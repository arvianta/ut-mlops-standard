from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def encode_payment_of_min_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies ordinal encoding to the 'payment_of_min_amount' column of the DataFrame.
    The 'payment_of_min_amount' column is assumed to contain binary values like 'No' and 'Yes',
    which will be encoded as integers.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'payment_of_min_amount' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'payment_of_min_amount' column ordinal encoded.
    """
    
    payment_min_order = [['no', 'yes']]  # Binary (No < Yes)
    
    ordinal_encoder = OrdinalEncoder(categories=payment_min_order, dtype=int, handle_unknown="use_encoded_value", unknown_value=-100)
    df['payment_of_min_amount'] = ordinal_encoder.fit_transform(df[['payment_of_min_amount']])
    
    return df
