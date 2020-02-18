"""Text-related util functions."""
from ftfy import fix_text


def clean(text: str, lowercase: bool = False) -> str:
    """Clean text.

    Does the following:
        1. Replace whitespace characters with space
        2. Fix corrupted text encodings
        3. Lowercase (if specified)
    
    Arguments:
        text {str} -- Text to clean
    
    Keyword Arguments:
        lowercase {bool} -- If True, perform lowercasing (default: {False})
    
    Returns:
        str -- Cleaned text
    """
    # Replace whitespace characters with spaces
    text = ' '.join(text.split())

    # Fix encoded texts
    text = fix_text(text)

    # Lowercase text if specified
    if lowercase:
        text = text.lower()

    return text
