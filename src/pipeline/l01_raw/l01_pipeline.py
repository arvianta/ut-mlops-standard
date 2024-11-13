from .node_databricks_ingest import databricks_ingest
from .node_gcs_ingest import gcs_ingest
from .node_bq_ingest import bq_ingest
from .node_utils import get_all_raw_files
from src.library.common import load_files_to_dict
from src.library.mlflow import log_features, log_artifact
from typing import Dict, Any


def run_layer_01(config: Dict[str, Any], **connections: Dict[str, Any]) -> list:
    """
    Executes data ingestion for Layer 1, loading data from specified sources
    (Databricks, Google Cloud Storage, BigQuery) based on configuration settings.
    The ingested data is saved as parquet files in the /data/raw directory.

    Args:
        config (Dict[str, Any]): Configuration settings, specifying ingestion sources and queries.
        **connections: Keyword arguments for source-specific client connections.

    Returns:
        list: File paths of ingested parquet files stored in /data/raw.
    """
    # Perform data ingestion from Databricks if enabled in config
    if config["connections"].get("databricks", {}).get("ingest", False) and "databricks" in connections:
        databricks_ingest(connections["databricks"])

    # Perform data ingestion from GCS if enabled in config
    if config["connections"].get("gcs", {}).get("ingest", False) and "gcs" in connections:
        gcs_ingest(connections["gcs"])

    # Perform data ingestion from BigQuery if enabled in config
    if config["connections"].get("bigquery", {}).get("ingest", False) and "bq" in connections:
        for table_name, query in config["connections"]["bigquery"].get("queries", {}).items():
            bq_ingest(query, connections["bq"], table_name)

    # Load all raw files into dataframes and log them to MLflow
    raw_file_paths = get_all_raw_files()
    raw_dataframes = load_files_to_dict(raw_file_paths)
    logged_features = log_features(raw_dataframes, "1_raw")
    log_artifact(logged_features[0], logged_features[1])
    
    return raw_file_paths
