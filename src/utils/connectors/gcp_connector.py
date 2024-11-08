import os
from google.cloud import storage, bigquery

def create_gcs_client(config):
    service_account_key_path = config["gcs"].get("service_account_key_path")
    gcs_client = storage.Client.from_service_account_json(service_account_key_path)
    return gcs_client


def create_bq_client(config):
    service_account_key_path = config["bigquery"].get("service_account_key_path")
    bq_client = bigquery.Client.from_service_account_json(service_account_key_path)
    return bq_client