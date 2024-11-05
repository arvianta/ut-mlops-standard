from src.connections.databricks_connector import create_databricks_client
import pandas as pd

def ingest_data_from_databricks(query, config):
    try:
        # Establish the Databricks connection using the reusable client function
        conn = create_databricks_client(config)
        
        # Execute query and load data into a DataFrame
        df = pd.read_sql(query, conn)
        
        # Close connection if necessary
        conn.close()
        
        return df
    except Exception as e:
        print(f"Error ingesting data from Databricks: {e}")
        raise