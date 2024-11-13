import pandas as pd
from src.library.common import standardize_column_names

def type_of_loan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary indicator columns for the nine most common loan types in the 'type_of_loan' column.
    Each new column represents a loan type, with values of `1` if that loan type is present in a row 
    and `0` otherwise. The original 'type_of_loan' column is then removed.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'type_of_loan' column with loan type information.

    Returns:
        pd.DataFrame: The modified DataFrame with binary columns for common loan types and without 
                      the original 'type_of_loan' column.
    
    Example:
        Given a 'type_of_loan' column with values ["Home", "Car", "Student, Car"], this function creates
        binary columns like `Home`, `Car`, etc., and drops 'type_of_loan'.
    """

    for loan_type in df['type_of_loan'].value_counts().head(9).index[1:]:
        df[loan_type] = df['type_of_loan'].str.contains(loan_type, na=False).astype(int)

    df.drop(columns=['type_of_loan'], inplace=True)
    
    df = standardize_column_names(df)
    
    return df
