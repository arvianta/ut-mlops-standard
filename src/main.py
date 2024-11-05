from src.connections.databricks_connector import create_databricks_client
from src.connections.gcp_connector import create_gcs_client, create_bq_client
from src.pipeline.pipeline import run_pipeline
import yaml

def load_config(config_path="config/config.yaml"):
    """Loads the project configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()

    # Initialize connections
    databricks_client = create_databricks_client(config)
    gcs_client = create_gcs_client(config)
    bq_client = create_bq_client(config)

    # Run the pipeline, passing in config and connections
    run_pipeline(config, databricks_client, gcs_client, bq_client)