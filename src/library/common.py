import os
import glob
import re
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

def save_dict_to_files(data: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Saves a dictionary of DataFrames to the specified directory, with each DataFrame
    saved as a .parquet file. The key in the dictionary will be used as the filename.

    Args:
        data (Dict[str, pd.DataFrame]): A dictionary of DataFrames to save.
        output_dir (str): The directory where the files will be saved.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each DataFrame in the dictionary as a .parquet file
    for file_name, df in data.items():
        file_path = os.path.join(output_dir, f"{file_name}.parquet")
        df.to_parquet(file_path, index=False)


def clear_yaml_files(directory: str) -> None:
    """
    Deletes all .yaml files in the specified directory.

    Args:
        directory (str): The directory from which to delete .yaml files.

    Returns:
        None
    """
    # Use glob to find all .yaml files in the directory
    yaml_files = glob.glob(os.path.join(directory, "*.yaml"))
    
    # Delete each .yaml file found
    for file_path in yaml_files:
        os.remove(file_path)
        print(f"Deleted: {file_path}")

    print(f"All .yaml files in {directory} have been cleared.")


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names by removing spaces, converting names to lowercase, 
    and replacing special characters with underscores.

    Args:
        df (pd.DataFrame): DataFrame with columns to be standardized.

    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """
    df.columns = [
        re.sub(r'[^0-9a-zA-Z]+', '_', col).strip('_').lower() for col in df.columns
    ]
    return df