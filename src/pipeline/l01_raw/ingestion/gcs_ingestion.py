import os
import pandas as pd
from src.connections.gcp_connector import create_gcs_client

def ingest_data_from_gcs(bucket_name, source_blob_name, destination_file_name, gcs_client):
    try:
        # Access the bucket and blob
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Download the blob to a local file
        blob.download_to_filename(destination_file_name)

        # Determine file type and load accordingly
        file_extension = os.path.splitext(destination_file_name)[1].lower()

        if file_extension == '.csv':
            df = pd.read_csv(destination_file_name)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(destination_file_name)
        elif file_extension == '.parquet':
            df = pd.read_parquet(destination_file_name)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return df
    except Exception as e:
        print(f"Error ingesting data from GCS: {e}")
        raise