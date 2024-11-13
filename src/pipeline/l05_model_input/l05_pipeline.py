from typing import Dict
import pandas as pd

from .node_knn_imputation import knn_imputation
from .node_split_data import split_data
from .node_scale_data import scale_data
from .node_smote import apply_smote
from src.library.common import save_dict_to_files
from src.library.mlflow import log_features, log_artifact

def run_layer_05(table: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
   """
   Processes data for the model input layer by performing feature selection, 
   imputation, outlier handling, and data splitting in preparation for training.

   Steps:
        1. **Feature Selection**: Identifies and retains the most relevant features 
           for the model to improve performance and interpretability.
        2. **Imputation**: Fills in missing values using appropriate methods 
           (e.g., mean, median, or mode imputation) to ensure model consistency.
        3. **Outlier Handling**: Detects and handles outliers, either by removing 
           or capping them, to avoid their impact on model performance.
        4. **Data Splitting**: Splits the data into training and test sets to evaluate 
           model generalization performance.

   Args:
        table (Dict[str, pd.DataFrame]): A dictionary containing a single entry 
            where the key is the table name, and the value is the DataFrame to process.

   Returns:
        Dict[str, pd.DataFrame]: A dictionary with the same structure as the input, 
        containing the processed DataFrame ready for model input.
   """
   # Processing steps (placeholder):
   # Implement feature selection, imputation, outlier handling, and data splitting here

   #Extract the single table from the dictionary
   df = table['train']
   target_column = "credit_score"
   
   # 1. Apply KNN imputation
   df = knn_imputation(df)
   
   # 1. Split the dataset into train and test
   x_train, x_test, y_train, y_test = split_data(df, target_column)

   # 2. Scale the data
   x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

   # 3. Apply SMOTE to handle class imbalance in training data
   x_train_smote, y_train_smote = apply_smote(x_train_scaled, y_train)

   # Combine X and y for both train and test sets into DataFrames
   train_combined = pd.concat([x_train_smote, y_train_smote], axis=1)
   test_combined = pd.concat([x_test_scaled, y_test], axis=1)

   # Update the table with the combined data
   table['train_combined'] = train_combined
   table['test_combined'] = test_combined
   
   # Save engineered features in the DataFrame to the feature data directory
   save_dict_to_files(table, "./data/5_model_input")
   
   # Log processed data to MLflow for tracking
   logged_features = log_features(table, "5_model_input")
   log_artifact(*logged_features)
   
   return table