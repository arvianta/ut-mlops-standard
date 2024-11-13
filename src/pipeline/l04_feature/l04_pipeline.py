from typing import Dict
import pandas as pd

from .node_type_of_loan import type_of_loan
from .node_encode_credit_score import encode_credit_score
from .node_encode_credit_mix import encode_credit_mix
from .node_encode_payment_of_min_amount import encode_payment_of_min_amount
from .node_encode_payment_behaviour import encode_payment_behaviour
from .node_encode_occupation import encode_occupation
from src.library.common import save_dict_to_files
from src.library.mlflow import log_features, log_artifact

def run_layer_04(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Executes Layer 4 of the data pipeline, focused on feature engineering. This layer receives a 
    dictionary containing a single table's DataFrame, applies various feature engineering techniques 
    to enhance the data, and returns the modified DataFrame within a dictionary structure.

    This layer performs tasks such as creating new features, transforming existing ones, encoding 
    categorical variables, and any other domain-specific feature engineering steps to make the data 
    more suitable for machine learning models.

    Args:
        tables (Dict[str, pd.DataFrame]): A dictionary containing one entry with the table name as 
                                          the key and its DataFrame as the value. This design allows 
                                          the function to identify the table by name.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary with the same table name as the key and the feature-
                                 engineered DataFrame as the value. This structure is retained for 
                                 consistency and to allow downstream tasks to refer to the table by name.
    
    Feature Engineering Steps:
        - **Feature Creation**: Generate new features based on domain knowledge or statistical properties.
        - **Transformation**: Apply transformations such as logarithmic, polynomial, or scaling as needed.
        - **Encoding**: Encode categorical variables using techniques like one-hot encoding or label encoding.
        - **Interaction Features**: Create interaction features if useful for model performance.
        - **Aggregation** (if applicable): Aggregate data over time windows or groups to create summary statistics.

    Notes:
        - This function assumes only one table is provided in the dictionary to avoid ambiguity.
        - The function can be extended to handle multiple tables if required in the future.
    
    Example:
        >>> processed_data = run_layer_04({"train": df})
        >>> print(processed_data["train"].head())  # View feature-engineered DataFrame.
    """
    
    # Extract the single table from the dictionary
    table_name, df = next(iter(tables.items()))
    
    # Apply type of loan transformations
    df = type_of_loan(df)
    
    # Apply all categorical encoding functions
    df = encode_credit_score(df)
    df = encode_credit_mix(df)
    df = encode_payment_of_min_amount(df)
    df = encode_payment_behaviour(df)
    df = encode_occupation(df)
    
    # Save engineered features in the DataFrame to the feature data directory
    table = {table_name: df}
    save_dict_to_files(table, "./data/4_feature")
    
    # Log processed data to MLflow for tracking
    logged_features = log_features(table, "4_feature")
    log_artifact(*logged_features)
    
    # Return the transformed DataFrame within the dictionary structure
    return table
