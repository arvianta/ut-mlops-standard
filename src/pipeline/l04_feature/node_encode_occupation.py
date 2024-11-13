import pandas as pd

def encode_occupation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies one-hot encoding to the 'occupation' column of the DataFrame.
    The 'occupation' column will be transformed into binary columns, one for each unique occupation.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'occupation' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'occupation' column one-hot encoded.
    """
    
    df = pd.get_dummies(df, columns=['occupation'], drop_first=True)
    
    return df
