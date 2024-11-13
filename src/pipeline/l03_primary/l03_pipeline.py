import pandas as pd
from typing import Dict
from .node_cleaning import clean_transform
from src.library.common import save_dict_to_files
from src.library.mlflow import log_features, log_artifact


def run_layer_03(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Executes Layer 3 of the data pipeline, performing data cleaning, joining, normalization, and
    transformation tasks to prepare data for downstream analysis or modeling. This layer processes 
    data by linking tables with primary and foreign keys, renaming columns to a consistent format, 
    normalizing data as needed, and creating a spine table if multiple tables are present.

    Args:
        tables (Dict[str, pd.DataFrame]): A dictionary where keys are table names and values are 
                                          DataFrames containing raw or intermediate data to be processed.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are table names and values are processed 
                                 DataFrames ready for further analysis or modeling.
    
    Steps:
        1. **Data Cleaning**: Apply basic cleaning procedures to each DataFrame, including handling 
           missing values, unusual data, and data type conversions as necessary.
        
        2. **Joining and Linking**: 
           - Identify and link tables based on primary and foreign key relationships.
           - If multiple tables are present, perform joins to combine related data.

        3. **Column Renaming**:
           - Rename columns to adhere to a consistent naming convention across all tables.
           - Ensure clarity and consistency in column names, e.g., snake_case or camelCase format.
        
        4. **Normalization**:
           - Normalize columns where appropriate, such as creating standardized units, encoding 
             categorical variables, or scaling numerical values.
        
        5. **Spine Table Creation** (if applicable):
           - If there are multiple tables, create a spine table that contains the primary keys and 
             other critical information necessary to serve as the base for further analysis or joining
             in later stages.

    Notes:
        - This function assumes that each DataFrame in the `tables` dictionary is structured with 
          identifiable primary and foreign keys where applicable.
        - Specific node functions (e.g., for data cleaning, joining, and renaming) should be developed 
          separately and called within this function to maintain modularity and reusability.

    Example:
        >>> cleaned_data = run_layer_03(raw_data)
        >>> print(cleaned_data["spine_table"])  # Inspect the spine table created in this layer.
    
    """
    # Check if the "train" table exists in the dictionary
    if "train" in tables:
        # Apply the cleaning and transformation function on the "train" table
        tables["train"] = clean_transform(tables["train"])
    
    # Save processed DataFrames to the primary data directory
    save_dict_to_files(tables, "./data/3_primary")
    
    # Log processed data to MLflow for tracking
    logged_features = log_features(tables, "3_primary")
    log_artifact(*logged_features)
    
    return tables
