import os
import mlflow
import pandas as pd
import yaml
from typing import Dict, Tuple


def config_mlflow(config) -> None:
    """
    Configures MLflow with the given parameters from the configuration file.

    Args:
        config (dict): Configuration dictionary containing MLflow settings.
    """
    mlflow_config = config["mlflow"]
    mode = mlflow_config["mode"]

    if mode == "local":
        mlflow.set_tracking_uri(mlflow_config["local"]["mlflow_tracking_uri"])
        mlflow.set_experiment(mlflow_config["local"]["experiment_name"])

    elif mode == "databricks":
        # Setting environment variables for Databricks authentication
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_config["databricks"]["mlflow_tracking_uri"]
        os.environ["DATABRICKS_HOST"] = mlflow_config["databricks"]["databricks_host"]
        os.environ["DATABRICKS_TOKEN"] = mlflow_config["databricks"]["databricks_token"]
        
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(mlflow_config["databricks"]["experiment_path"])
        

    else:
        raise ValueError(f"Unsupported MLflow mode: {mode}")

    print(f"MLflow configured in {mode} mode.")


def start_mlflow_run(run_name: str) -> None:
    """
    Starts an MLflow run with a given name. 
    If a run is already active, it will continue to use the existing run.
    
    Args:
        run_name (str): The name of the MLflow run.
        
    Returns:
        mlflow.ActiveRun: The active MLflow run object.
    """
    if not mlflow.active_run():
        mlflow.start_run(run_name=run_name)
        print(f"Started new MLflow run with name: {run_name}")
    else:
        print(f"Using existing MLflow run with ID: {mlflow.active_run().info.run_id}")

    return mlflow.active_run()


def log_features(
    data_dict: Dict[str, pd.DataFrame],
    phase: str,
) -> Tuple[str, str]:
    """
    Logs the feature names and basic info of multiple DataFrames into a phase_features.yaml file.

    Args:
        data_dict (Dict[str, pd.DataFrame]): Dictionary where keys are table names and values are DataFrames.
        phase (str): The phase of the data pipeline (e.g., raw, intermediate, etc.).
    """
    
    # Prepare the path for the YAML file
    log_dir = os.path.join(os.getcwd(), "log")  # Absolute path to 'log' directory
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    
    file_name = f"{phase}_features.yaml"
    file_path = os.path.join(log_dir, file_name)
    artifact_path = "data"
    
    # Load existing YAML data if the phase_features.yaml file already exists
    try:
        with open(file_path, 'r') as file:
            existing_data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        existing_data = {}
        
    # Loop over each table in the dictionary and log its feature info
    for table_name, data in data_dict.items():
        # Extract feature names and data types
        feature_data_types = data.dtypes.apply(str).to_dict()
        
        feature_info = {
            'table_info': {
                'data_types': feature_data_types,
                'table_name': table_name,
                'feature_numbers': len(data.columns),
                'row_number': len(data)
            }
        }
        
        # Append the new dataset feature info to the existing data
        existing_data[table_name] = feature_info

    # Write updated data back to YAML file
    with open(file_path, 'w') as file:
        yaml.dump(existing_data, file, default_flow_style=False)

    print(f"Feature information for all tables saved to {file_path}")
    
    # Return the file path and artifact path
    return artifact_path, file_path


def log_artifact(
    artifact_path: str, 
    file_to_log: str
) -> None:
    """
    Logs a specified file as an artifact to the current MLflow experiment run.

    Args:
        artifact_path (str): The path within the MLflow artifact store where the file will be saved.
        file_to_log (str): The local file path of the file to log.
    """
    try:
        # Check if there's an active run
        active_run = mlflow.active_run()

        if active_run:
            # Log artifact directly if there's an active run
            mlflow.log_artifact(file_to_log, artifact_path)
            print(f"File '{file_to_log}' successfully logged to artifact path '{artifact_path}' in active run ID: {active_run.info.run_id}")
        else:
            # Start a new run if none is active
            with mlflow.start_run() as run:
                mlflow.log_artifact(file_to_log, artifact_path)
                print(f"File '{file_to_log}' successfully logged to artifact path '{artifact_path}' in new run ID: {run.info.run_id}")
    
    except Exception as e:
        print(f"Failed to log artifact '{file_to_log}' to path '{artifact_path}': {e}")
