import os
from google.cloud import storage, bigquery

def create_gcs_client(credentials_path):
    try:
        client = storage.Client.from_service_account_json(credentials_path)
        return client
    except Exception as e:
        print(f"Error creating GCS client: {e}")
        raise

def create_bq_client(credentials_path):
    try:
        client = bigquery.Client.from_service_account_json(credentials_path)
        return client
    except Exception as e:
        print(f"Error creating BigQuery client: {e}")
        raise