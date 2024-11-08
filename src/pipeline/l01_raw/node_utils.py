import os

def get_all_raw_files():
    """
    Retrieves all parquet file paths from the /data/raw directory.

    Returns:
        list: A list of file paths for all files in the /data/raw directory.
    """
    raw_dir = '/data/raw'
    return [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.parquet')]