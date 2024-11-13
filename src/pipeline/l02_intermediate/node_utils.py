import pandas as pd
from typing import Dict, List, Optional, Union

META_COL = "cols_changed"

def clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names.

    Args:
        df: Input dataframe

    Returns:
        pd.DataFrame: Cleaned up dataframe
    """
    df.columns = [str(col).lower().strip() for col in df.columns]  # type: ignore[assignment]
    return df


def _get_column_order(
    primary_keys: List[str], desc_by: List[str], all_columns: List[str]
) -> List[str]:
    # primary_keys and desc_by must be the original list because converting to set does not guarantee order

    other_value_columns = sorted(set(all_columns) - (set(primary_keys) | set(desc_by)))
    # convert primary_keys and desc_by to set just to get the columns that aren't included in primary_keys or desc_by
    # then sort the other value columns. this will be appended for sorting
    return primary_keys + desc_by + other_value_columns


def deduplicate(
    data: pd.DataFrame,
    primary_keys: Optional[List[str]] = None,
    desc_by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """If primary keys are provided, then take only 1 row for each key, else deduplicates by all columns.

    Deduplication using `primary_keys` will first sort using all columns, where column order
        is determined by `primary_keys + desc_by + remaining columns`. After sorting,
        only the first row is kept for each primary key.

    Args:
        data: input data to undergo deduplication.
        primary_keys: the column names of the primary keys
        desc_by: the names of columns to rank on, usually a update time/create time

    Returns:
        deduplicated data
    """
    if primary_keys:
        if desc_by is None:
            desc_by = []
        all_columns = data.columns.to_list()
        column_order = _get_column_order(primary_keys, desc_by, all_columns)
        not_null_data = data.dropna(axis=0, subset=primary_keys)

        # make sure prioritize not null value, regardless of sort_values
        unique_data = not_null_data.sort_values(
            column_order, ascending=False, na_position="last"
        ).drop_duplicates(  # to be safe, user needs to specify all columns to desc by
            subset=primary_keys, keep="first"
        )
    else:
        unique_data = data.drop_duplicates()

    return unique_data.reset_index(drop=True)


def suggest_attributes_for_processing(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
) -> Dict[str, Optional[Union[Dict[str, str], List[str], str]]]:
    """
    Suggests the attributes needed for the `standard_processing` function
    based on common data patterns in the given DataFrame or dictionary of DataFrames.
    
    Args:
        data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): Input data.
        
    Returns:
        Dict[str, Optional[Union[Dict[str, str], List[str], str]]]: Dictionary of suggested attributes.
    """
    # If the data is a dictionary, concatenate it into a single DataFrame
    if isinstance(data, dict):
        data = pd.concat(data.values(), axis=0, ignore_index=True)
    
    # Dictionary to store the suggested attributes
    attributes = {}
    
    # Suggest data types based on common types and patterns in columns
    dtype = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            dtype[col] = "float64" if data[col].dtype == 'float' else "int64"
        elif pd.api.types.is_datetime64_any_dtype(data[col]):
            dtype[col] = "datetime64[ns]"
        else:
            dtype[col] = "string"
    attributes["dtype"] = dtype
    
    # Suggest column renaming based on common naming issues (e.g., whitespace)
    col_names = {col: col.strip().lower().replace(" ", "_") for col in data.columns if " " in col or col.isupper()}
    attributes["col_names"] = col_names if col_names else None
    
    # Primary keys: suggest any column ending in 'id' or 'ID' as primary keys
    primary_keys = [col for col in data.columns if "id" in col.lower()]
    attributes["primary_keys"] = primary_keys if primary_keys else None
    
    # Order by columns: suggest columns related to dates or timestamps
    order_by = [col for col in data.columns if "date" in col.lower() or "time" in col.lower()]
    attributes["order_by"] = order_by if order_by else None
    
    # Partition key: if any column resembles a partitioning column, suggest it
    # E.g., "partition", "partition_key", etc.
    partition_key = next((col for col in data.columns if "partition" in col.lower()), None)
    attributes["partition_key"] = partition_key
    
    # Usecols: suggest keeping all columns initially, though this can be modified as needed
    attributes["usecols"] = list(data.columns)
    
    # Date formats: attempt to infer formats for columns in datetime dtype or containing 'date'
    dt_format = {}
    for col in data.columns:
        if "date" in col.lower() and pd.api.types.is_string_dtype(data[col]):
            # Infer date format if possible (for simplicity, assuming "%Y%m%d" pattern)
            dt_format[col] = "%Y%m%d"
    attributes["dt_format"] = dt_format if dt_format else None
    
    return attributes