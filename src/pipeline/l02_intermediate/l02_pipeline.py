from .node_standard_processing import standard_processing
from .node_utils import suggest_attributes_for_processing
from src.library.common import load_files_to_dict, save_dict_to_files
from src.library.mlflow import log_features, log_artifact
from typing import Dict
import pandas as pd


def run_layer_02(raw_paths: list) -> Dict[str, pd.DataFrame]:
    """
    Executes Layer 2 of the pipeline by loading raw data files, processing each DataFrame,
    and saving the processed data into the intermediate data directory. The processed data
    is logged to MLflow for tracking and versioning.

    Args:
        raw_paths (list): A list of file paths to raw data files.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are table names and values are processed DataFrames.
    """
    # Load raw data files into a dictionary of DataFrames
    data = load_files_to_dict(raw_paths)
    
    # Suggest attributes for processing based on data analysis
    attributes_to_process = suggest_attributes_for_processing(data)
    
    # Inspect suggested attributes for verification
    print("Suggested attributes for processing:", attributes_to_process)
    print("Type of 'attributes_to_process':", type(attributes_to_process))
    
    # Initialize a dictionary to store processed DataFrames
    processed_data = {}
    for table_name, df in data.items():
        # Process each DataFrame using standard_processing
        processed_df = standard_processing(df, attributes_to_process)
        processed_data[table_name] = processed_df
    
    # Save processed DataFrames to the intermediate data directory
    save_dict_to_files(processed_data, "./data/2_intermediate")
    
    # Log processed data to MLflow for tracking
    logged_features = log_features(processed_data, "2_intermediate")
    log_artifact(*logged_features)
    
    return processed_data