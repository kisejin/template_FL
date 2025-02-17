import os
import json

def clean_output_text(text):
    """
    Clean and normalize text from LLM outputs by removing noise and repetitions.
    
    Args:
        text (str): Raw text from LLM prediction
        
    Returns:
        str: Cleaned and normalized text
    """
    import re
    
    def remove_repeats(text):
        # Remove repeated words
        pattern_words = r'\b(\w+)(?:\s+\1\b)+'
        text = re.sub(pattern_words, r'\1', text)

        # Remove repeated character patterns (like 'asasas')
        pattern_chars = r'(\w+?)\1+'
        text = re.sub(pattern_chars, r'\1', text)

        return text
    
    # Remove excessive punctuation
    def normalize_punctuation(text):
        # Replace multiple exclamation/question marks with single ones
        text = re.sub(r'!+', '!', text)
        text = re.sub(r'\?+', '?', text)
        # Remove multiple periods (except for ellipsis)
        text = re.sub(r'\.{4,}', '...', text)
        text = text.replace('cor', '').replace('asesa', '')
        return text
    
    # Main cleaning pipeline
    cleaned_text = text.strip()
    
    # Remove common noise patterns
    noise_patterns = [
        r'\n+',              # Multiple newlines
        r'\s+',              # Multiple spaces
        r'\\n',              # Literal \n
        r'\\t',              # Literal \t
    ]
    
    for pattern in noise_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text)
    
    # Apply cleaning functions
    # cleaned_text = remove_repetitions(cleaned_text)
    cleaned_text = remove_repeats(cleaned_text)
    cleaned_text = normalize_punctuation(cleaned_text)
    cleaned_text = ' '.join(cleaned_text.split())  # Normalize spacing
    
    return cleaned_text.strip()

def save_client_metrics(client_id: int, round_number: int, metrics: dict, folder="result_metric/20250217_045516"):
    """
    Save or update a JSON file for a client with metrics from the given round.
    The file will be named client-{client_id}.json and follow the format:
    
    {
        "round_{i}": {
            "f1": <value>,
            "rouge1": <value>
        },
        ...
    }
    """
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"client-{client_id}.json")
    
    # Load previous metrics if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}
    
    # Update data with current round metrics
    data[f"round_{round_number}"] = metrics
    
    # Write the updated data back to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def save_server_metrics(round_number: int, task_metrics: dict, folder="result_metric/20250217_045516"):
    """
    Save or update a JSON file for the server with metrics from the given round.
    The file will be named global_server.json and follow the format:
    
    {
        "round_{i}": {
            "task1": {
                "f1": <value>,
                "rouge1": <value>
            },
            "task2": {
                "f1": <value>,
                "rouge1": <value>
            },
            ...
        },
        ...
    }
    """
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, "global_server.json")
    
    # Load previous metrics if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {}
    
    # Update data with current round metrics
    data[f"round_{round_number}"] = task_metrics
    
    # Write the updated data back to file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)