import pandas as pd
from imblearn.over_sampling import SMOTE
from typing import Tuple

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_seed: int = 101) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance the class distribution
    in the training set by generating synthetic samples.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training labels.
        random_seed (int): Random seed for reproducibility (default is 101).

    Returns:
        pd.DataFrame: Resampled training features.
        pd.Series: Resampled training labels.
    """
    smote = SMOTE(sampling_strategy='auto', random_state=random_seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled
