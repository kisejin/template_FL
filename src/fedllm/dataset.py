from trl import DataCollatorForCompletionOnlyLM

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

FDS = None  # Cache FederatedDataset

def split_train_test(dataset, test_size):
    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(dataset.to_pandas(), test_size=test_size, shuffle=True, random_state=42)

    # Convert to Dataset objects
    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    # Combine into a DatasetDict
    datasets_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset.select(range(10))
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
    print(client_trainset)
    return client_trainset

def load_data1(partition_id: int, num_partitions: int, dataset_name: str):
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
    # client_trainset = client_trainset.rename_column("output", "response")
    client_set = split_train_test(client_trainset, test_size=0.2)
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