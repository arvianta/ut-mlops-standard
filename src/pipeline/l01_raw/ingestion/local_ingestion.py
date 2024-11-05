import pandas as pd
import os

def ingest_data_local(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    # Determine file type and load accordingly
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    elif file_extension == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return df