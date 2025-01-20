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