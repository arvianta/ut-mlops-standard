from pyspark.sql import SparkSession
import os

def databricks_ingest(query: str, spark: SparkSession) -> str:
    """Ingests data from Databricks via Spark, saves it as parquet, and returns the file path."""
    try:
        # Run query and get the Spark DataFrame
        df = spark.sql(query)
        
        # Define the raw data path and save as parquet
        raw_data_path = "/data/raw/databricks_data.parquet"
        df.write.mode("overwrite").parquet(raw_data_path)
        
        print(f"Data from Databricks saved to {raw_data_path}")
        return raw_data_path
    
    except Exception as e:
        print(f"Error ingesting data from Databricks: {e}")
        raise
