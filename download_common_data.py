import gdown
from pathlib import Path

file_id = '1WVwH1A2huPwNBQ8xN4KN0APq8oCg_ct9'

def _gdrive_url(file_id):
    """Convert Google Drive file ID to direct download URL."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

if Path('common_data').exists() or Path('common_data.tar.gz').exists():
    pass
else:
    gdown.download(_gdrive_url(file_id), './')

