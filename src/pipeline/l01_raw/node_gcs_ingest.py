import os
import pandas as pd
from google.cloud import storage

def gcs_ingest(client: storage.Client, files: dict) -> dict:
    """
    Ingests specified files from Google Cloud Storage, saves them as parquet files in /data/raw,
    and returns a dictionary of file paths.

    Args:
        client (storage.Client): Google Cloud Storage client instance.
        files (dict): Dictionary where keys are labels for the files (e.g., "file_1") and values
                      are tuples in the format (bucket_name, source_blob_name). Each source_blob_name
                      should include the full path to the file in GCS (e.g., "path/to/file.csv").

    Returns:
        dict: Dictionary where keys are the file labels and values are the paths to the saved parquet files.
    """
    data_paths = {}
    
    for label, (bucket_name, source_blob_name) in files.items():
        try:
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            
            # Extract the original file name and define paths
            file_name = os.path.basename(source_blob_name)
            temp_file_path = f"/tmp/{file_name}"
            raw_data_path = f"/data/raw/{os.path.splitext(file_name)[0]}.parquet"
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)  # Ensure the directory exists
            
            # Download file temporarily to /tmp
            blob.download_to_filename(temp_file_path)
            
            # Determine file extension and load data into a DataFrame
            file_extension = os.path.splitext(file_name)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(temp_file_path)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(temp_file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(temp_file_path)
            else:
                print(f"Unsupported file format for {file_name}. Skipping.")
                os.remove(temp_file_path)
                continue
            
            # Save the DataFrame as a parquet file in /data/raw
            df.to_parquet(raw_data_path, index=False)
            data_paths[label] = raw_data_path
            
            # Clean up temporary file
            os.remove(temp_file_path)
            print(f"File {file_name} from GCS saved to {raw_data_path}")
        
        except Exception as e:
            print(f"Error ingesting file {label} from GCS: {e}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    return data_paths
