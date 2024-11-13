from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def encode_payment_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies ordinal encoding to the 'payment_behaviour' column of the DataFrame.
    The 'payment_behaviour' column contains multiple categories that will be encoded as integers.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'payment_behaviour' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'payment_behaviour' column ordinal encoded.
    """
    
    payment_behaviour_order = [
        'low_spent_small_value_payments',
        'low_spent_medium_value_payments',
        'low_spent_large_value_payments',
        'high_spent_small_value_payments',
        'high_spent_medium_value_payments',
        'high_spent_large_value_payments'
    ]
    
    ordinal_encoder = OrdinalEncoder(categories=[payment_behaviour_order], dtype=int, handle_unknown="use_encoded_value", unknown_value=-100)
    df['payment_behaviour'] = ordinal_encoder.fit_transform(df[['payment_behaviour']])
    
    return df
