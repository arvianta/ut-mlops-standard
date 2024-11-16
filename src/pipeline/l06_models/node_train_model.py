import mlflow
import os
import tensorflow as tf

from typing import Tuple, Dict, Any
from .node_utils import evaluate_model, save_model_config

def train_model(X_train, Y_train, X_test, Y_test, tuner, epochs=50, batch_size=32) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Dict[str, float], Dict[str, Any]]:
    """Train the model using the best hyperparameters and evaluate it."""
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    # Get hyperparameters as dictionary
    hyperparameters = {
        'dropout_1': best_hps.get('dropout_1'),
        'dropout_2': best_hps.get('dropout_2'),
        'dropout_3': best_hps.get('dropout_3'),
        'learning_rate': best_hps.get('learning_rate'),
        'epochs': epochs,
        'batch_size': batch_size
    }

    final_model_name = f"BEST_MODEL_d{hyperparameters['dropout_1']}_{hyperparameters['dropout_2']}_{hyperparameters['dropout_3']}_lr{hyperparameters['learning_rate']}"

    with mlflow.start_run(run_name=final_model_name, nested=True) as run:
        # Log hyperparameters
        mlflow.log_params(hyperparameters)

        # Training callback
        class MLflowTrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    mlflow.log_metrics({
                        "training_balanced_accuracy": logs.get('balanced_accuracy', 0),
                        "training_accuracy": logs.get('accuracy', 0),
                        "training_loss": logs.get('loss', 0),
                        "epoch": epoch
                    }, step=epoch)

        # Train the model
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[MLflowTrainingCallback()]
        )

        # Evaluate the model
        metrics, confusion_matrix_path = evaluate_model(model, X_test, Y_test, run.info.run_id)
        
        # Log evaluation metrics
        mlflow.log_metrics(metrics)
        
        # Log confusion matrix plot
        mlflow.log_artifact(confusion_matrix_path)
        
        # Save and log model configuration
        config_path = save_model_config(hyperparameters, metrics)
        mlflow.log_artifact(config_path)
        
        # Save the model directly with mlflow.tensorflow
        mlflow.tensorflow.log_model(model, artifact_path="model")

        # Clean up temporary files
        os.remove(confusion_matrix_path)
        os.remove(config_path)

    return model, history, metrics, hyperparameters