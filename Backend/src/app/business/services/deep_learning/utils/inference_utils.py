import os

def ensure_dir(path):
    """Ensure directory exists for a file path."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)