from typing import Tuple
import pandas as pd
import os

from tensorflow.keras.utils import to_categorical
from .node_tune_model import tune_model
from .node_train_model import train_model

def run_layer_06(data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]) -> str:
    """
    This function processes the data from the training and testing stages and returns the trained final model.

    **Parameters:**
    - `data_tuple` (tuple): A tuple containing the training and testing data:
      - `x_train` (np.ndarray): The input features for the training data.
      - `y_train` (np.ndarray): The target labels for the training data.
      - `x_test` (np.ndarray): The input features for the test data.
      - `y_test` (np.ndarray): The target labels for the test data.

    **Returns:**
    - `model` (tensorflow.keras.Model): The trained model after training and hyperparameter tuning.
    
    **Usage Example:**
    ```python
    model = run_layer_06((x_train, y_train, x_test, y_test))
    ```
    """

    tuning_epochs = 10
    final_epochs = 50
    batch_size = 32
    
    x_train, y_train, x_test, y_test = data
    
    Y_train = to_categorical(y_train, num_classes=3)
    Y_test = to_categorical(y_test, num_classes=3)
    
    tuner = tune_model(x_train, Y_train, max_trials=5, epochs=tuning_epochs, batch_size=batch_size)
    
    model, history, metrics, hyperparameters = train_model(
        x_train, Y_train, 
        x_test, Y_test, 
        tuner, 
        epochs=final_epochs, 
        batch_size=batch_size
    )
    
    model_name = 'credit_score_model_tuned.h5'
    
    # Save the model locally
    model_dir = './data/6_model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the Keras model
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)

    return model
