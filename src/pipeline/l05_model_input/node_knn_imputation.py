from sklearn.impute import KNNImputer
import pandas as pd

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


def knn_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs K-Nearest Neighbors (KNN) imputation on specified columns to fill missing values.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns with missing values.

    Returns:
        pd.DataFrame: The DataFrame with imputed values in specified columns.
    
    Columns Imputed:
        - 'num_of_delayed_payment'
        - 'num_credit_inquiries'
        - 'credit_history_age'
        - 'amount_invested_monthly'
    """
    knn_impute_column(df, 'num_of_delayed_payment')
    knn_impute_column(df, 'num_credit_inquiries')
    knn_impute_column(df, 'credit_history_age')
    knn_impute_column(df, 'amount_invested_monthly')
    
    return df
