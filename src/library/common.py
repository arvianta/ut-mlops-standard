import os
import pandas as pd
from typing import Union, List, Dict

def load_files_to_dict(file_paths: Union[str, List[str]]) -> Dict[str, pd.DataFrame]:
    """
    Reads one or multiple files (CSV, Excel, TXT, Parquet) from the provided path(s) and 
    returns a dictionary with file names as keys and DataFrames as values.

    Args:
        file_paths (Union[str, List[str]]): A file path or a list of file paths to read.
    
    Returns:
        Dict[str, pd.DataFrame]: A dictionary with each file name (without extension) as a key
                                 and the corresponding DataFrame as the value.
    """
    # Ensure file_paths is a list for uniform processing
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    dataframes = {}
    
    for file_path in file_paths:
        # Get file name without extension for the dictionary key
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            # Load the file into a DataFrame based on its extension
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            elif file_extension == '.txt':
                df = pd.read_csv(file_path, delimiter='\t')  # Assuming tab-delimited for .txt
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Add DataFrame to the dictionary with file name as key
            dataframes[file_name] = df
        
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return dataframes
