import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scales the feature set using MinMaxScaler to bring all features within the range [0, 1].

    Parameters:
        X_train (pd.DataFrame): Training features to be scaled.
        X_test (pd.DataFrame): Testing features to be scaled.

    Returns:
        pd.DataFrame: Scaled training features.
        pd.DataFrame: Scaled testing features.
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled
