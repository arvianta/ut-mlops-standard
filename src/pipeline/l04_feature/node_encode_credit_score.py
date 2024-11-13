from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def encode_credit_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies ordinal encoding to the 'credit_score' column of the DataFrame.
    The 'credit_score' column is assumed to contain values like 'Poor', 'Standard', and 'Good',
    which will be encoded as integers based on the predefined order.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'credit_score' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'credit_score' column ordinal encoded.
    """
    
    # Define the order of categories for Credit_Score
    credit_score_order = [['poor', 'standard', 'good']]  # Ordered from lowest to highest
    
    # Create the OrdinalEncoder for Credit_Score
    ordinal_encoder = OrdinalEncoder(categories=credit_score_order, dtype=int)
    
    # Apply ordinal encoding to the 'credit_score' column
    df['credit_score'] = ordinal_encoder.fit_transform(df[['credit_score']])
    
    return df
