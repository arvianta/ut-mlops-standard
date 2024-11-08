import os
import pandas as pd
from google.cloud import bigquery

def bq_ingest(query: str, bq_client: bigquery.Client, table_name: str) -> str:
    """
    Ingests data from BigQuery based on a query, saves it as a parquet file in /data/raw
    with the table name as the filename, and returns the file path.

    Args:
        query (str): The SQL query to execute in BigQuery.
        bq_client (bigquery.Client): BigQuery client instance.
        table_name (str): The name of the BigQuery table or dataset, used for naming the output file.

    Returns:
        str: The path to the saved parquet file.
    """
    try:
        # Execute the query and load results into a DataFrame
        df = bq_client.query(query).to_dataframe()
        
        # Define the file path with the original table name
        raw_data_path = f"/data/raw/{table_name}.parquet"
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)  # Ensure the directory exists
        
        # Save the DataFrame as a parquet file
        df.to_parquet(raw_data_path, index=False)
        
        print(f"Data from BigQuery saved to {raw_data_path}")
        return raw_data_path
    
    except Exception as e:
        print(f"Error ingesting data from BigQuery: {e}")
        raise
