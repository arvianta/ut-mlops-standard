import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_seed: int = 101) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets based on the target column.

    Parameters:
        df (pd.DataFrame): DataFrame containing features and target.
        target_column (str): The name of the target column for prediction.
        test_size (float): Proportion of data to use for testing (default is 0.2).
        random_seed (int): Random seed for reproducibility (default is 101).

    Returns:
        pd.DataFrame: Training feature set.
        pd.DataFrame: Testing feature set.
        pd.Series: Training labels.
        pd.Series: Testing labels.
    """
    X = df.drop(columns=target_column)
    y = df[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    return X_train, X_test, y_train, y_test