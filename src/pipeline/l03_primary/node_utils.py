import numpy as np
import re
from sklearn.impute import KNNImputer

def history_age_month(age: str) -> int:
    """
    Converts a credit history age string (e.g., 'X Years and Y Months') into total months.

    Args:
        age (str): Credit history duration in the format 'X Years and Y Months'.

    Returns:
        int: Total months of credit history or NaN if input is invalid.
    """
    try:
        years = int(re.findall(r'\d+', age.split("and")[0])[0])
        months = int(re.findall(r'\d+', age.split("and")[1])[0])
        return years * 12 + months
    except (IndexError, ValueError, AttributeError):
        return np.nan


def knn_impute_column(df, column, n_neighbors=5):
    """
    Impute missing values in the specified column using KNN.
    Args: df (DataFrame), column (str), n_neighbors (int): Number of neighbors (Default is 5).
    Returns: DataFrame: DataFrame with imputed column.
    """
    # Apply KNN imputation to the specified column
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[[column]] = imputer.fit_transform(df[[column]])
    
    return df