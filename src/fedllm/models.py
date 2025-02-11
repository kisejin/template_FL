import math

import torch
import torch.nn as nn
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainerCallback, 
    # BertForSequenceClassification,
    BertConfig,
)

from .skipbert.modeling import BertForSequenceClassification, SkipBertForSequenceClassification

from flwr.common.typing import NDArrays
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from thop import profile
import wandb
from typing import Dict, List
import copy
import time
import numpy as np


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """
    use_cuda = torch.cuda.is_available()
    device_map = torch.device("cuda:0" if use_cuda else "cpu")
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif model_cfg.quantization == 0:
        quantization_config = None
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization or 0-bit for no quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        # torch_dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if model_cfg.flash_attention else "eager"
        ),
    ).to(device_map)
    
    if use_cuda:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )
    
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=model_cfg.lora.lora_r,
        lora_alpha=model_cfg.lora.lora_alpha,
        lora_dropout=model_cfg.lora.lora_dropout,
        target_modules=model_cfg.lora.lora_target_modules.split(", "),
        bias="none",
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config), tokenizer



def get_custom_config(teacher_name, skipbert_args: DictConfig):

    num_labels = 1 # Set number of labels to 1 for regression or single-class tasks
    teacher_config = BertConfig.from_pretrained(teacher_name)
    teacher_config.num_labels = num_labels
    teacher_config.fit_size = teacher_config.hidden_size
    student_config = BertConfig.from_pretrained(skipbert_args.student_model)
    student_config.num_labels = num_labels
    student_config.fit_size = teacher_config.hidden_size


    if skipbert_args.num_layers_student > 0:
        student_config.num_hidden_layers = skipbert_args.num_layers_student

    if skipbert_args.num_full_hidden_layers_student > 0:
        student_config.num_full_hidden_layers = skipbert_args.num_full_hidden_layers_student

    else:
        student_config.num_full_hidden_layers = student_config.num_hidden_layers

    student_config.task_type = skipbert_args.output_mode
    student_config.n_gram_left = skipbert_args.n_gram_left
    student_config.n_gram_right = skipbert_args.n_gram_right
    #     student_config.plot_mode = 'plot_passive'
    student_config.plot_mode = 'force_compute'
    student_config.ngram_masking = 0.

    if not hasattr(student_config, 'enter_hidden_size'):
        student_config.enter_hidden_size = student_config.hidden_size

    if not hasattr(student_config, 'max_num_entries'):
        student_config.max_num_entries = 100000

    return teacher_config, student_config



def get_data_influence_model(model_cfg: DictConfig, skipbert_args: DictConfig):
    use_cuda = torch.cuda.is_available()
    device_map = torch.device("cuda" if use_cuda else "cpu")

    # Load model with num_labels=1
    teacher_name = "bert-base-uncased"

    teacher_config, student_config = get_custom_config(teacher_name=teacher_name, skipbert_args=skipbert_args)

    
    # Load model with num_labels=1
    teacher_model = BertForSequenceClassification.from_pretrained(
        teacher_name, config=teacher_config
    ).to(device_map)

    

    student_model = SkipBertForSequenceClassification.from_pretrained(
        skipbert_args.student_model, config=student_config, 
        do_fit=skipbert_args.do_fit, 
        # share_param=skipbert_args.share_param
    ).to(device_map)



    if skipbert_args.freeze_lower_layers:
        student_model.freeze_shallow_layers()

    tokenizer = AutoTokenizer.from_pretrained(
        teacher_name, do_lower_case=skipbert_args.do_lower_case, use_fast=True
    )
    
    if use_cuda:
        teacher_model = prepare_model_for_kbit_training(
            teacher_model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )
        
        student_model = prepare_model_for_kbit_training(
            student_model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )

    return teacher_model, student_model, tokenizer


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]

def model_parameters_to_ndarrays(model):
    """
    Convert the parameters of a HuggingFace model into a list of NDArrays.

    Args:
        model (torch.nn.Module): The HuggingFace model.

    Returns:
        list[NDArrays]: A list of NumPy arrays representing the model's parameters.
    """
    ndarrays = []
    for param_tensor in model.state_dict().values():
        # Convert PyTorch tensor to NumPy array
        ndarrays.append(param_tensor.cpu().numpy())
    return ndarrays


def concatenate_models_with_marker(main_model_params: list[NDArrays], 
                                   data_influence_model_params: list[NDArrays],
                                   marker_value: float = np.nan) -> list[NDArrays]:
    """
    Concatenate two models' parameters with a unique marker.

    Args:
        main_model_params (list[NDArrays]): Parameters of the main model as NDArrays.
        data_influence_model_params (list[NDArrays]): Parameters of the data influence model as NDArrays.
        marker_value (float): A unique marker value to separate the two models.

    Returns:
        list[NDArrays]: A single list of NDArrays with the unique marker separating the models.
    """
    marker = np.array([marker_value])  # Unique marker
    concatenated_params = main_model_params + [marker] + data_influence_model_params
    return concatenated_params


def split_models(concatenated_model: list[NDArrays]) -> tuple[list[NDArrays], list[NDArrays]]:
    """Split the concatenated model back into main and data influence models."""
    # Find the marker's index
    marker_index = next(
        (i for i, param in enumerate(concatenated_model) if np.isnan(param).all()),
        -1,
    )
    if marker_index == -1:
        raise ValueError("Marker not found in the concatenated model parameters.")

    main_model = concatenated_model[:marker_index]
    data_influence_model = concatenated_model[marker_index + 1 :]
    return main_model, data_influence_model


def set_parameters_bert(model: BertForSequenceClassification, parameters: list[NDArrays]) -> None:
    """
    Set the parameters of a BertForSequenceClassification model using the given ones.

    Args:
        model (BertForSequenceClassification): The model whose parameters need to be updated.
        parameters (list[NDArrays]): A list of NumPy arrays representing the parameters.
    """
    # Get the state_dict keys from the model
    state_dict_keys = model.state_dict().keys()
    
    # Ensure the number of parameters matches the model's state_dict
    if len(parameters) != len(state_dict_keys):
        raise ValueError(
            f"Number of parameters ({len(parameters)}) does not match "
            f"the number of state_dict keys ({len(state_dict_keys)})."
        )
    
    # Create an OrderedDict to update the model
    params_dict = zip(state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    
    # Load the updated state_dict into the model
    model.load_state_dict(state_dict)
