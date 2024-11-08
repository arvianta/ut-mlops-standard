import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from .node_utils import clean_col_names, deduplicate, META_COL

def standard_processing(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    dtype: Optional[Dict[str, str]] = None,
    col_names: Optional[Dict[str, str]] = None,
    primary_keys: Optional[List[str]] = None,
    order_by: Optional[List[str]] = None,
    partition_key: Optional[str] = None,
    usecols: Optional[List[str]] = None,
    dt_format: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Basic steps for generating intermediate layer data.

    1. rename columns
    2. remove fully empty records
    3. trim string
    4. empty string casted to np.nan
    5. convert type
    6. lower string
    7. remove duplicates

    Args:
        data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): Raw data.
        dtype (Optional[Dict[str, str]]): Keys are new column names, values are data types.
            Must be valid data types acceptable by astype
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html.
        col_names (Optional[Dict[str, str]]): Keys are original column names,
            values are new column names.
        primary_keys (Optional[List[str]]): The primary key names.
        order_by (Optional[List[str]]): The columns to order by for deduplication,
            usually update or create time.
        partition_key (Optional[str]): The partition column that does not count towards actual data.
        usecols (Optional[List[str]]): List of new column names to keep.
        dt_format (Optional[Dict[str, str]]): Keys are new column names,
            values are datetime format e.g. '%Y%m%d'.

    Returns:
        pd.DataFrame: processed data
    """

    def _pk_should_have_null(df: pd.DataFrame) -> pd.DataFrame:
        if primary_keys:
            return df.dropna(subset=primary_keys, how="any").reset_index(drop=True)
        return df

    # Concatenate data if it is dictionary, following the order of the data keys/names
    if isinstance(data, dict):
        data = pd.concat(
            [df for _, df in sorted(data.items())], axis=0, ignore_index=True
        )

    partition_key = partition_key or ""

    # expect to read in everything as string columns
    # renames, case insensitive
    renamed_data = clean_col_names(
        data.drop(columns=[META_COL], errors="ignore").drop_duplicates()
    )
    if col_names:
        renamed_data = renamed_data.rename(
            columns={str(k).lower(): v for k, v in col_names.items()}
        )

    # drop records that are all null
    data_cols = [
        col for col in renamed_data.columns if col not in {partition_key, META_COL}
    ]
    not_null_data = renamed_data.dropna(subset=data_cols, how="all")

    # trim & strip any 'object' column
    # cast any empty string value to np.nan
    str_types = ["string", "object"]

    trim_data = not_null_data.copy()
    all_cols = [
        col
        for col in trim_data.select_dtypes(include=str_types).columns
        if col != META_COL
    ]
    trim_data[all_cols] = (
        trim_data[all_cols]
        .astype(str)
        .applymap(lambda x: x.strip().strip("'").strip('"'), na_action="ignore")
        .applymap(lambda x: np.nan if x in ["", "nan"] else x, na_action="ignore")
    )
    # keep subset of columns
    if usecols:
        trim_data = trim_data[usecols]
        all_cols = usecols

    # convert type
    # all lower for string
    typed_data = _pk_should_have_null(trim_data)

    if dtype:
        for col, d in dtype.items():
            if d in {"datetime64[ns]"}:
                if dt_format and dt_format.get(col):
                    typed_data[col] = pd.to_datetime(
                        typed_data[col], errors="coerce", format=dt_format[col]
                    )
                else:
                    typed_data[col] = pd.to_datetime(
                        typed_data[col], errors="coerce", infer_datetime_format=True
                    )
            elif d.lower() in {"float64", "int64"}:
                typed_data[col] = pd.to_numeric(typed_data[col], errors="coerce")

        typed_data = typed_data.astype(dtype)
        str_columns = typed_data[all_cols].select_dtypes(include=str_types).columns
        typed_data[str_columns] = typed_data[str_columns].applymap(
            lambda x: x.lower() if pd.notna(x) else x
        )

    # deduplicate
    unique_data = deduplicate(_pk_should_have_null(typed_data), primary_keys, order_by)

    return unique_data