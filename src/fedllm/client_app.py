"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig

from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
from trl import SFTTrainer, SFTConfig

from .dataset import (
    get_data_collator_and_propt_formatting,
    load_data,
    load_data_homo,
    load_data_hete,
    replace_keys,
)
from .models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)

from .flwr_mods import get_wandb_mod
from .metrics import exact_match, f1, get_rouge_score
from .utils import clean_output_text
from .make_data import Prompter, generate_and_tokenize_prompt

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


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
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        
        self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
        # self.training_argumnets = SFTConfig(**train_cfg.training_arguments, max_seq_length=train_cfg.seq_length) 
        
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
        
    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = 1829
        pred_ids = np.argmax(pred.predictions, axis=-1)
        # all unnecessary tokens are removed
        pred_str = self.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = self.tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True
        )
        return {
            **get_rouge_score(predictions=pred_str, targets=label_str),
            # **exact_match(predictions=pred_str, targets=label_str),
            **f1(predictions=pred_str, targets=label_str),
        }
    
    def _make_dataset(self):
        prompter = Prompter(self.train_cfg.prompt_template_name, self.train_cfg.verbose)
        tmp_dict = {
            "prompter": prompter,
            "seq_length": self.train_cfg.seq_length,
            "train_on_inputs": self.train_on_inputs,
            "tokenizer": self.tokenizer,
        }

        self.trainset = (
            self.trainset
            .shuffle()
            .map(
                lambda x: generate_and_tokenize_prompt(x, **tmp_dict),
                num_proc=8,
            )
        )

        self.valset = (
            self.valset
            .shuffle()
            .map(
                lambda x: generate_and_tokenize_prompt(x, **tmp_dict),
                num_proc=8,
            )
        )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]

        # Construct supervised trainer
        # trainer = SFTTrainer(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     args=self.training_argumnets,
        #     train_dataset=self.trainset,
        #     eval_dataset=self.valset,
        #     formatting_func=self.formatting_prompts_func,
        #     data_collator=self.data_collator,
        #     compute_metrics=self.compute_metrics,
        # )
        
        # Constuct baseline Trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=self.trainset,
            eval_dataset=self.valset.select(range(10)),
            args=self.training_argumnets,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Do local training
        results = trainer.train()

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Let's get the client partition
    if cfg.dataset.type == 'homo':
        client_set = load_data_homo(partition_id, num_partitions, cfg.dataset.name)
    else:
        client_set = load_data_hete(partition_id)

    return FlowerClient(
        cfg.model,
        cfg.train,
        client_set['train'],
        client_set['test'],
        num_rounds,
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
    mods=[
        get_wandb_mod("FL@CSS25"),
    ],
)