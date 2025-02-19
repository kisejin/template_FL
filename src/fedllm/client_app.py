"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

import torch
import logging
import wandb
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from .utils import save_client_metrics
from omegaconf import DictConfig


from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    EarlyStoppingCallback,
    # BertForSequenceClassification,
    GenerationConfig,
)

from trl import SFTTrainer, SFTConfig
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator

from .trainer import ManualTrainer

from .dataset import (
    get_data_collator_and_propt_formatting,
    load_data,
    load_data_homo,
    load_data_hete,
    replace_keys,
)
from .models import *

from .flwr_mods import get_wandb_mod
from .metrics import exact_match, f1, get_rouge_score
from .utils import clean_output_text, save_client_metrics
from .make_data import Prompter, generate_and_tokenize_prompt
from .server_app import datetime_str

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


logging.getLogger("flwr").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


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
        max_length=seq_len,
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
        mates_args: DictConfig,
        skipbert_args: DictConfig,
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
        self.mates_args = mates_args
        self.skipbert_args = skipbert_args
        self.holdoutset = None
        self.refset = None
        self.teacher_data_influence_model = None
        self.student_data_influence_model = None
        self.data_influence_tokenizer = None

        # instantiate model
        self.model, self.tokenizer = get_model(model_cfg)

        if self.mates_args.state:
            (
                self.teacher_data_influence_model,
                self.student_data_influence_model,
                self.data_influence_tokenizer,
            ) = get_data_influence_model(model_cfg, skipbert_args)

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

    def compute_metrics(self, pred):
        labels_ids = pred["label_ids"]
        pred_ids = pred["predictions"]

        # Replace -100 with pad token id in labels
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id

        print(f"Shape of predictions: {np.shape(pred_ids)}")
        print(f"Shape of labels: {np.shape(labels_ids)}")

        # Decode predictions and labels
        pred_str = self.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = self.tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True
        )

        # Remove any extra whitespace from the decoded strings
        pred_str = [s.strip() for s in pred_str]
        label_str = [s.strip() for s in label_str]

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

        # Process trainset
        self.trainset = self.trainset.shuffle().map(
            lambda x: generate_and_tokenize_prompt(x, **tmp_dict),
            num_proc=8,
            remove_columns=self.trainset.column_names,
        )

        # Process valset
        self.valset = self.valset.shuffle().map(
            lambda x: generate_and_tokenize_prompt(x, **tmp_dict),
            num_proc=8,
            remove_columns=self.valset.column_names,
        )

        # Create holdoutset and refset if state is True
        if self.mates_args.state:
            trainset_size = len(self.trainset)

            # Calculate sizes for holdout and reference sets
            holdout_size = int(trainset_size * self.mates_args.holdout_ratio)
            ref_size = int(trainset_size * self.mates_args.reference_ratio)

            # Shuffle the trainset to ensure randomness
            shuffled_indices = list(range(trainset_size))
            self.trainset = self.trainset.shuffle()

            # Split the dataset
            holdout_indices = shuffled_indices[:holdout_size]
            ref_indices = shuffled_indices[
                holdout_size : holdout_size + ref_size + 1
            ]

            # Create holdoutset and refset
            self.holdoutset = self.trainset.select(holdout_indices)
            self.refset = self.trainset.select(ref_indices)

            print(
                f"Holdoutset size: {len(self.holdoutset)}, Refset size: {len(self.refset)}"
            )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        selection_fraction = 1.0
        """Implement distributed fit function for a given client."""

        if self.mates_args.state and int(config["current_round"]) != 1:
            main_model_params, data_influence_model_params = split_models(
                parameters
            )
            set_parameters(self.model, main_model_params)
            set_parameters_bert(
                self.teacher_data_influence_model, data_influence_model_params
            )

            # Compute the total number of tokens in the training set.
            # print(self.tokenizer.decode(self.trainset[0]['input_ids'], skip_special_tokens = True))
            total_tokens = sum(
                len(
                    f"{self.tokenizer.decode(sample['input_ids'], skip_special_tokens = True)}".split()
                )
                for sample in self.trainset
            )  # adjust tokenizer if needed

            # Compute the total number of parameters in the main model.
            # main_model_param_count = sum(param.numel() for param in main_model_params) # Pytorch params
            main_model_param_count = sum(
                param.size for param in main_model_params
            )  # Numpy params
            print(
                f"Total tokens: {total_tokens}, Total params: {main_model_param_count}\n"
            )

            # Calculate the optimal number of training tokens based on the Chinchilla scaling law.
            D_opt = self.mates_args.tokens_per_param * main_model_param_count
            selection_fraction = D_opt / total_tokens
            selection_fraction = min(selection_fraction, 1.0)
        else:
            set_parameters(self.model, parameters)

            # Calculate the optimal number of training tokens based on the Chinchilla scaling law
            D_opt = self.mates_args.tokens_per_param * len(parameters)
            selection_fraction = D_opt / len(self.trainset)
            selection_fraction = min(selection_fraction, 1.0)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = config["save_path"]

        # Initialize callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5
        )

        # Construct supervised trainer
        # trainer = SFTTrainer(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     args=self.training_arguments,
        #     train_dataset=self.trainset,
        #     eval_dataset=self.valset,
        #     formatting_func=self.formatting_prompts_func,
        #     data_collator=self.data_collator,
        #     compute_metrics=self.compute_metrics,
        #     callbacks=[flops_callback, early_stopping_callback]
        # )

        # # Constuct baseline Trainer
        # trainer = Trainer(
        #     model=self.model,
        #     train_dataset=self.trainset,
        #     eval_dataset=self.valset.select(range(10)),
        #     args=self.training_arguments,
        #     data_collator=self.data_collator,
        #     compute_metrics=self.compute_metrics,
        #     callbacks=[early_stopping_callback]
        # )

        trainer = ManualTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.trainset,
            val_dataset=self.valset.select(range(10)),
            holdout_dataset=self.holdoutset,
            reference_dataset=self.refset,
            args=self.training_arguments,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            mates_args=self.mates_args,
            skipbert_args=self.skipbert_args,
            selection_fraction=selection_fraction,
            teacher_data_influence_model=self.teacher_data_influence_model,
            student_data_influence_model=self.student_data_influence_model,
            data_influence_tokenizer=self.data_influence_tokenizer,
            task="text-generation",
        )

        # Train the model
        results = trainer.train()

        if self.mates_args.state:
            # After training
            main_model_params = get_parameters(self.model)
            data_influence_model_params = model_parameters_to_ndarrays(
                self.teacher_data_influence_model
            )
            final_model_params = concatenate_models_with_marker(
                main_model_params, data_influence_model_params
            )
        else:
            final_model_params = get_parameters(self.model)

        torch.cuda.empty_cache()

        # Calculate FLOPs
        with get_accelerator().device("cuda"):
            batch_size = self.training_arguments.per_device_eval_batch_size
            seq_len = self.train_cfg.seq_length

            flops1, macs1, params1 = get_model_profile(
                self.model,
                kwargs=input_constructor(batch_size, seq_len, self.tokenizer),
                print_profile=True,
                detailed=False,
            )

            flops2, macs2, params2 = get_model_profile(
                self.teacher_data_influence_model,
                kwargs=input_constructor(
                    batch_size, seq_len, self.data_influence_tokenizer
                ),
                print_profile=True,
                detailed=False,
            )

            flops1_value, flops2_value = convert_to_float(
                flops1
            ), convert_to_float(flops2)
            macs1_value, macs2_value = convert_to_float(
                macs1
            ), convert_to_float(macs2)
            params1_value, params2_value = convert_to_float(
                params1
            ), convert_to_float(params2)

            wandb.log(
                {
                    "total_flops": flops1_value + flops2_value,
                    "macs": macs1_value + macs2_value,
                    "params": params1_value + params2_value,
                }
            )

        print_results = {
            "train_loss": results["training_loss"],
            "flops": flops1_value + flops2_value,
            "eval_loss": results["eval_loss"],
        }

        # Save results to filde

        save_client_metrics(
            client_id=self.id,
            round_number=int(config["current_round"]),
            metrics={
                **print_results,
                **results["eval_scores"],
                "total_flops": flops1_value + flops2_value,
            },
            folder=f"result_metric/{datetime_str}",
        )
        return (
            final_model_params,
            len(self.trainset),
            print_results,
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

    cfg.skipbert.att_layer_maps = [
        int(s) for s in cfg.skipbert.att_layer_maps.split(", ")
    ]
    cfg.skipbert.hid_layer_maps = [
        int(k) for k in cfg.skipbert.hid_layer_maps.split(", ")
    ]

    return FlowerClient(
        cfg.model,
        cfg.train,
        cfg.mates,
        cfg.skipbert,
        client_set["train"],
        client_set["test"],
        num_rounds,
        partition_id,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
    mods=[
        get_wandb_mod("FL@CSS25_skipbert_mates"),
    ],
)
