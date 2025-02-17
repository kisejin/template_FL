"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from .dataset import (
    get_data_collator_and_propt_formatting,
    load_data,
    load_data_hete,
    load_data_homo,
    replace_keys,
)
from .flwr_mods import get_wandb_mod
from .make_data import Prompter, generate_and_tokenize_prompt
from .metrics import exact_match, f1, get_rouge_score
from .models import cosine_annealing, get_model, get_parameters, set_parameters
from .server_app import datetime_str
from .utils import clean_output_text, save_client_metrics

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


def input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(
        seq_len - 2
    ):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    inputs = tokenizer(
        [fake_seq] * batch_size,
        padding=True,
        truncation=True,
        # max_length=seq_len,
        return_tensors="pt",
    )
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    # inputs.update({"labels": torch.unsqueeze(labels,dim=0)})

    # To device
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    return inputs


def convert_to_float(value_str):
    value, unit = value_str.split()
    value = float(value)
    if unit == "T" or "T" in unit:
        return value * 1e12
    elif unit == "G" or "G" in unit:
        return value * 1e9
    elif unit == "M" or "M" in unit:
        return value * 1e6
    elif unit == "K" or "K" in unit:
        return value * 1e3
    return value


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        valset,
        num_rounds,
        client_id,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.train_cfg = train_cfg
        self.id = client_id

        self.training_arguments = TrainingArguments(
            **train_cfg.training_arguments
        )
        # self.training_arguments = SFTConfig(**train_cfg.training_arguments, max_seq_length=train_cfg.seq_length)

        self.num_rounds = num_rounds
        self.trainset = trainset
        self.valset = valset

        # instantiate model
        self.model, self.tokenizer = get_model(model_cfg)

        # (
        #     self.data_collator,
        #     self.formatting_prompts_func
        # ) = get_data_collator_and_propt_formatting(self.tokenizer)

        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )

        self.train_on_inputs = self.train_cfg.train_on_inputs

        self._make_dataset()

        self.train_loader = DataLoader(
            self.trainset,
            batch_size=self.training_arguments.per_device_train_batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.data_collator,
            drop_last=self.train_cfg.training_arguments.dataloader_drop_last,
        )

        self.val_loader = DataLoader(
            self.valset,
            batch_size=self.training_arguments.per_device_eval_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.data_collator,
        )

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_arguments.learning_rate,
        )

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_rounds,
        )

        # Adding accelerator
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        )

    def mytrain(self):
        self.model.train()
        total_loss = 0
        for i, batch in tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            mininterval=self.train_cfg.training_arguments.logging_steps,
            bar_format="{l_bar}{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient accumulation if needed
            if (
                (i + 1)
                % self.train_cfg.training_arguments.gradient_accumulation_steps
                == 0
            ):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            if (i + 1) % self.train_cfg.training_arguments.logging_steps == 0:
                print(f"Batch {i}, Step {i+1}, Loss: {loss.item():.4f}")
                wandb.log({"Train_loss": loss.item()})
        wandb.log({"Total_train_loss": total_loss / len(self.train_loader)})
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def myevaluate(self) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        self.model.eval()
        total_loss = 0.0
        all_predictions, all_labels = [], []

        for i, batch in tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            mininterval=self.train_cfg.training_arguments.logging_steps,
            bar_format="{l_bar}{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            loss = outputs.loss

            predictions = outputs.logits.argmax(dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            total_loss += loss.item()

        metrics = self.compute_metrics(all_predictions, all_labels)
        metrics["eval_loss"] = total_loss / len(self.val_loader)
        wandb.log(metrics)
        return metrics

    def compute_metrics(self, predictions, labels):
        labels_ids = labels
        labels_ids[labels_ids == -100] = 1829
        print("labels_ids", labels_ids)
        print("predictions", predictions)
        # pred_ids = np.argmax(predictions, axis=-1)
        pred_ids = predictions
        # all unnecessary tokens are removed
        pred_str = self.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = self.tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True
        )
        return {
            **get_rouge_score(predictions=pred_str, targets=label_str),
            **f1(predictions=pred_str, targets=label_str),
        }

    def _make_dataset(self):
        prompter = Prompter(
            self.train_cfg.prompt_template_name, self.train_cfg.verbose
        )
        tmp_dict = {
            "prompter": prompter,
            "seq_length": self.train_cfg.seq_length,
            "train_on_inputs": self.train_on_inputs,
            "tokenizer": self.tokenizer,
        }

        self.trainset = self.trainset.shuffle().map(
            lambda x: generate_and_tokenize_prompt(x, **tmp_dict),
            num_proc=8,
            remove_columns=self.trainset.column_names,
        )

        self.valset = self.valset.shuffle().map(
            lambda x: generate_and_tokenize_prompt(x, **tmp_dict),
            num_proc=8,
            remove_columns=self.valset.column_names,
        )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        # Update learning rate cosine annealing
        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        # Training loop
        self.training_arguments.output_dir = config["save_path"]

        # Do local training
        num_epochs = self.train_cfg.training_arguments.num_train_epochs
        train_loss = 0.0
        print("Training...")
        for epoch in range(int(num_epochs)):
            train_loss = self.mytrain()
            print(f"Epoch {epoch}, Train_loss: {train_loss:.4f}")
            wandb.log({"Train_loss": train_loss})

        # Do local evaluation
        print("Evaluating...")
        eval_results = self.myevaluate()
        wandb.log(eval_results)
        print(f"Evaluation metrics: {eval_results:.4f}")

        # Calculate FLOPs
        with get_accelerator().device("cuda"):
            batch_size = self.training_arguments.per_device_eval_batch_size
            seq_len = self.train_cfg.seq_length
            flops, macs, params = get_model_profile(
                self.model,
                kwargs=input_constructor(batch_size, seq_len, self.tokenizer),
                print_profile=True,
                detailed=False,
            )
            flops_value = convert_to_float(flops)
            macs_value = convert_to_float(macs)
            params_value = convert_to_float(params)
            wandb.log(
                {
                    "total_flops": flops_value,
                    "macs": macs_value,
                    "params": params_value,
                }
            )  # wa

        results_metrics = {
            "train_loss": train_loss,
            "eval_loss": eval_results["loss"],
            "total_flops": flops_value,
            "eval_f1": eval_results["f1"],
            "eval_rouge1": eval_results["rouge1"],
            "eval_rouge2": eval_results["rouge2"],
            "eval_rougeL": eval_results["rougeL"],
            "eval_rougeLsum": eval_results["rougeLsum"],
        }

        # Save client metrics
        save_client_metrics(
            client_id=self.id,
            round_number=config["current_round"],
            metrics=results_metrics,
            folder=f"result_metric/{datetime_str}",
        )

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": train_loss, "flops": flops_value},
        )


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    if cfg.dataset.type == "homo":
        client_set = load_data_homo(
            partition_id, num_partitions, cfg.dataset.name
        )
    else:
        client_set = load_data_hete(partition_id)

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_set["train"],
        client_set["test"],
        num_rounds,
        partition_id,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
    mods=[
        get_wandb_mod("FL@CSS25_baseline"),
    ],
)
