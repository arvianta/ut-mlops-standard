from databricks_api import DatabricksAPI
import yaml
import os

def load_databricks_config():
    config_path = "config/databricks_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_databricks_client():
    config = load_databricks_config()
    host = config.get("host")
    token = config.get("token")

    if not host or not token:
        raise ValueError("Databricks host and token must be provided in the config file.")

    # Initialize DatabricksAPI client
    client = DatabricksAPI(
        host=host,
        token=token
    )
    return client