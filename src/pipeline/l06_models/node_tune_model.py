import keras_tuner as kt
import pandas as pd
import mlflow

from .node_utils import TuningMLflowCallback, build_model

def tune_model(X_train: pd.DataFrame, Y_train: pd.Series, max_trials: int = 5, epochs: int = 10, batch_size: int = 32) -> kt.Hyperband:
    """
    Tunes a model using the Hyperband algorithm with specified hyperparameters
    and logs parameters to an MLflow nested run.

    Parameters:
        X_train (Any): Training features, typically as a NumPy array or Pandas DataFrame.
        Y_train (Any): Training labels, typically as a NumPy array or Pandas Series.
        max_trials (int): Maximum number of trials for the tuner to run. Default is 5.
        epochs (int): Number of epochs for each trial. Default is 10.
        batch_size (int): Batch size for model training. Default is 32.

    Returns:
        kt.Hyperband: An instance of the Hyperband tuner after performing the search.
    """
    # Start an MLflow nested run
    with mlflow.start_run(run_name="hyperparameter-tuning", nested=True):
        # Log hyperparameter tuning settings
        mlflow.log_params({
            "max_trials": max_trials,
            "tuning_epochs": epochs,
            "batch_size": batch_size
        })
        
        # Initialize Hyperband tuner with the model-building function
        tuner = kt.Hyperband(
            build_model,
            objective=kt.Objective('balanced_accuracy', direction='max'),
            max_epochs=epochs,
            factor=3,
            hyperband_iterations=1,
            directory='./log/tuner_dir',
            project_name='credit_score_tuning'
        )

        # Custom tuner class to integrate MLflow logging callback
        class MyTuner(kt.Hyperband):
            def run_trial(self, trial: kt.engine.trial.Trial, *args, **kwargs):
                kwargs['callbacks'] = kwargs.get('callbacks', []) + [
                    TuningMLflowCallback(trial.trial_id, trial.hyperparameters.values)
                ]
                return super().run_trial(trial, *args, **kwargs)
        
        # Instantiate and run MyTuner
        tuner = MyTuner(
            hypermodel=build_model,
            objective=kt.Objective('balanced_accuracy', direction='max'),
            max_epochs=epochs,
            factor=3,
            hyperband_iterations=1,
            directory='./log/tuner_dir',
            project_name='credit_score_tuning'
        )

        # Perform hyperparameter search
        tuner.search(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    
    return tuner