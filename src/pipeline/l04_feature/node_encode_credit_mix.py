from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def encode_credit_mix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies ordinal encoding to the 'credit_mix' column of the DataFrame.
    The 'credit_mix' column is assumed to contain values like 'Bad', 'Standard', 'Good', 'Unknown',
    which will be encoded as integers based on the predefined order.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'credit_mix' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'credit_mix' column ordinal encoded.
    """
    
    credit_mix_order = [['bad', 'standard', 'good', 'unknown']]  # Ordered from lowest to highest
    
    ordinal_encoder = OrdinalEncoder(categories=credit_mix_order, dtype=int)
    df['credit_mix'] = ordinal_encoder.fit_transform(df[['credit_mix']])
    
    return df
