from trl import DataCollatorForCompletionOnlyLM

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd


FDS = None  # Cache FederatedDataset
client_id_ds = None
global_test_set_homo = None

def split_train_test(dataset, test_size):
    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(dataset.to_pandas(), test_size=test_size, shuffle=True, random_state=42)
    test_data, global_test = train_test_split(test_data, test_size=0.1, shuffle=True, random_state=42)

    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)
    global_test = Dataset.from_pandas(global_test)

    # Combine into a DatasetDict
    datasets_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return datasets_dict



def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_data_collator_and_propt_formatting(tokenizer):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return data_collator, formatting_prompts_func


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    print(f"<---- Load client {partition_id} --->")
    client_trainset = FDS.load_partition(partition_id, "train")
    client_trainset = client_trainset.rename_column("output", "response")
    return client_trainset

def load_data_homo(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    global global_test_set_homo
    if FDS is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
        # list_ds = []
        # for cid in range(0,num_partitions):
        #     tmp_set = FDS.load_partition(cid, "train")
        #     list_ds.append(
        #         pd.DataFrame(tmp_set)
        #     )
        # list_ds = pd.concat(list_ds, ignore_index=True)
        # _, global_test_set_homo = train_test_split(
        #         list_ds, test_size=0.1, shuffle=True, random_state=42
        # )
        # global_test_set_homo = Dataset.from_pandas(global_test_set_homo).remove_columns(['__index_level_0__'])
                    
    print(f"<---- Load client {partition_id} --->")
    client_trainset = FDS.load_partition(partition_id, "train")
    # client_trainset = client_trainset.rename_column("output", "response")
    client_set = split_train_test(client_trainset, test_size=0.2)
    return client_set


def load_data_hete(partition_id: int):
    """Load partition data heterogeneous"""
    global client_id_ds
    if client_id_ds is None:
        from .data_domains import client_id_dataset
        client_id_ds = client_id_dataset
    print(f"<---- Load client {partition_id} --->")
    client_set = client_id_ds[str(partition_id)]
    return client_set


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict