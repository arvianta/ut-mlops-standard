import pandas as pd
import mlflow
import yaml

def log_features (
    data: pd.DataFrame, 
    phase: str, 
    table_name: str,
    ) -> None:
    
    """
    Logs the feature names and basic info of a DataFrame into a phase_features.yaml file.

    Args:
        df: The DataFrame itself.
        dataset_name: From which layer the dataframe belongs to.
        table_name: The name of the table.
    """
    
    # Extract feature names and data types
    feature_data_types = data.dtypes.apply(str).to_dict()
    
    feature_info = {
        'table_info': {
            'table_name': table_name,
            'feature_numbers': len(data.columns),
            'row_number': len(data.rows),
            'data_types': feature_data_types
        }
    }
    
    file_name = "../../log/%s_features.yaml" % phase
    
    # Load existing YAML data if phase_features.yaml already exists
    try:
        with open(file_name, 'r') as file:
            existing_data = yaml.safe_load(file) or {}
    except FileNotFoundError:
        existing_data = {}
        
    # Append the new dataset feature info to the existing data
    existing_data.update(feature_info)

    # Write updated data back to YAML file
    with open(file_name, 'w') as file:
        yaml.dump(existing_data, file, default_flow_style=False)

    print(f"Feature information for {table_name} saved to {file_name}")

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