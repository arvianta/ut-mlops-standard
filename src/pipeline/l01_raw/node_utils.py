import os
from typing import List

def get_all_raw_files() -> List[str]:
    """
    Retrieves all parquet file paths from the /data/1_raw directory.

    Returns:
        list: A list of file paths for all files in the /data/1_raw directory.
    """
    # Absolute path construction
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the root directory of the project
    raw_dir = os.path.join(base_dir, '../../data/1_raw')
    
    # Check if the directory exists, to avoid the FileNotFoundError
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"The directory {raw_dir} does not exist.")

    # Create a list of file paths for all parquet files
    raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.parquet')]
    
    # Return the list of raw files
    return raw_files