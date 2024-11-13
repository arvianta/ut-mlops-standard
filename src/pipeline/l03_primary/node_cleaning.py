import numpy as np
import pandas as pd
import re

from .node_utils import history_age_month

def clean_transform(train: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and transforms a financial dataset for modeling. Operations include:
    - Handling missing values and imputing data based on logical assumptions
    - Normalizing and converting data types
    - Creating binary columns for categorical variables
    - Standardizing numeric fields by removing non-numeric characters
    - Dropping unnecessary columns

    Parameters:
        train (pd.DataFrame): The input DataFrame containing raw training data.

    Returns:
        pd.DataFrame: The cleaned and transformed DataFrame ready for further analysis.
    """
    
    # Annual Income
    train['annual_income'] = train['annual_income'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x))).astype(float)
    
    # Monthly Inhand Salary (Fill missing using annual income)
    train['monthly_inhand_salary'].fillna(train['annual_income'] / 12, inplace=True)
    
    train['credit_history_age'] = train['credit_history_age'].apply(history_age_month)

    # Num of Delayed Payment
    train['num_of_delayed_payment'] = train['num_of_delayed_payment'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x))).replace('', np.nan).astype(float).abs()

    # Amount Invested Monthly
    train['amount_invested_monthly'] = train['amount_invested_monthly'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x))).replace('', np.nan).astype(float).abs()
    
    # Monthly Balance
    train['monthly_balance'] = train['monthly_balance'].replace('__-333333333333333333333333333__', np.nan)
    train['monthly_balance'] = train['monthly_balance'].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else x)
    train['monthly_balance'] = train['monthly_balance'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x))).replace('', np.nan)
    train['monthly_balance'] = pd.to_numeric(train['monthly_balance'], errors='coerce').abs()
    train['monthly_balance'] = train.groupby('delay_from_due_date')['monthly_balance'].transform(lambda x: x.fillna(x.mean()))
    
    # Age (Remove unrealistic values)
    train['age'] = pd.to_numeric(train['age'].apply(lambda x: re.sub(r'[^0-9.]+', '', str(x))), errors='coerce')
    train.loc[(train['age'] > 70) | (train['age'] < 18), 'age'] = np.nan
    train['age'] = train.groupby(['credit_history_age', 'delay_from_due_date'])['age'].transform(lambda x: x.fillna(x.mean()))
    train['age'].fillna(train['age'].mean(), inplace=True)

    # Occupation (Replace unknowns)
    train['occupation'].replace('_______', 'unknown', inplace=True)
    
    # Num Bank Accounts (Set invalid values to zero)
    train['num_bank_accounts'].replace(-1, 0, inplace=True)

    # Num of Loan (Standardize and clean)
    train['num_of_loan'] = train['num_of_loan'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x))).replace('', np.nan)
    train['num_of_loan'] = pd.to_numeric(train['num_of_loan'], errors='coerce').abs()
    
    # Delay from Due Date (Convert negatives to positives)
    train['delay_from_due_date'] = train['delay_from_due_date'].abs()

    # Changed Credit Limit (Standardize format)
    train['changed_credit_limit'] = train['changed_credit_limit'].apply(lambda x: re.sub(r'[^0-9.-]+', '', str(x))).replace('', 0).astype(float)

    # Credit Mix (Replace missing or unknown values)
    train['credit_mix'].replace('_', 'unknown', inplace=True)

    # Outstanding Debt
    train['outstanding_debt'] = pd.to_numeric(train['outstanding_debt'].apply(lambda x: re.sub(r'[^0-9.-]+', '', str(x))), errors='coerce')

    # Payment Behaviour (Handle unusual values)
    train['payment_behaviour'].replace('!@9#%8', np.nan, inplace=True)
    train = train[train['payment_behaviour'] != 'unknown']

    # Payment of Min Amount (Handle unusual values)
    train['payment_of_min_amount'].replace('NM', np.nan, inplace=True)
    train = train[train['payment_of_min_amount'] != 'unknown']
    
    # Payment of Min Amount (Handle unusual values)
    train['payment_of_min_amount'].replace('nm', np.nan, inplace=True)
    train = train[train['payment_of_min_amount'] != 'unknown']
    
    # Drop unnecessary columns
    columns_to_drop = ['id', 'customer_id', 'name', 'ssn', 'month']
    train.drop(columns=columns_to_drop, inplace=True)

    return train
