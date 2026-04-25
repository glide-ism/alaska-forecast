import gdown
from pathlib import Path

file_id = '0'

def _gdrive_url(file_id):
    """Convert Google Drive file ID to direct download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

if Path('common_data').exists() or Path('common_data.tar.gz').exists():
    pass
else:
    gdown.download(_gdrive_url(file_id), './')

