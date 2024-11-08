from .node_databricks_ingest import databricks_ingest
from .node_gcs_ingest import gcs_ingest
from .node_bq_ingest import bq_ingest
from .node_utils import get_all_raw_files

def run_layer_01(config, **connections):
    """
    Executes Layer 1 of the pipeline, ingesting data from configured sources (Databricks, GCS, BigQuery, Local).
    Each data source saves its output as parquet files in the /data/raw directory.

    Args:
        config (dict): Configuration dictionary with ingestion settings for each data source.
        **connections: Variable keyword arguments for client connections to external data sources.

    Returns:
        list: A list of file paths for all ingested parquet files in /data/raw.
    """
    # Databricks ingestion if configured
    if config["connections"].get("databricks", {}).get("ingest", False) and "databricks" in connections:
        databricks_ingest(connections["databricks"])

    # GCS ingestion if configured
    if config["connections"].get("gcs", {}).get("ingest", False) and "gcs" in connections:
        gcs_ingest(connections["gcs"])

    # BigQuery ingestion if configured
    if config["connections"].get("bigquery", {}).get("ingest", False) and "bq" in connections:
        bq_query_configs = config["connections"]["bigquery"].get("queries", {})
        for table_name, query in bq_query_configs.items():
            bq_ingest(query, connections["bq"], table_name)

    # Retrieve all ingested file paths from /data/raw
    return get_all_raw_files()                      
