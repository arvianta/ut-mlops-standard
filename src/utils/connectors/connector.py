from .databricks_connector import create_databricks_client
from .gcp_connector import create_gcs_client, create_bq_client

def create_connection(config):
    connections = {}
    
    # Initialize Databricks client if enabled
    if config.get("connections", {}).get("databricks", {}).get("enabled", False):
        databricks_config = config["connections"]["databricks"]
        connections["databricks_client"] = create_databricks_client(databricks_config)
    
    # Initialize GCS client if enabled and verify service account key path
    if config.get("connections", {}).get("gcs", {}).get("enabled", False):
        gcs_config = config["connections"]["gcs"]
        if not gcs_config.get("service_account_key_path"):
            raise ValueError("GCS service account key path is required but not provided in config.")
        connections["gcs_client"] = create_gcs_client(gcs_config)
    
    # Initialize BigQuery client if enabled and verify service account key path
    if config.get("connections", {}).get("bigquery", {}).get("enabled", False):
        bigquery_config = config["connections"]["bigquery"]
        if not bigquery_config.get("service_account_key_path"):
            raise ValueError("BigQuery service account key path is required but not provided in config.")
        connections["bq_client"] = create_bq_client(bigquery_config)
    
    # Local configuration
    if config.get("connections", {}).get("local", {}).get("enabled", False):
        local_config = config["connections"]["local"]
        connections["local_config"] = local_config

    return connections
