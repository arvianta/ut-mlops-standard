from src.utils.config import load_config
from src.utils.connectors.connector import create_connection
from src.pipeline.pipeline import run_pipeline

def main():
    # Load configuration
    config = load_config()

    # Determine ingestion method and initialize required connection(s)
    connections = create_connection(config)
    
    # Run the pipeline, passing in config and connections
    run_pipeline(config, **connections)

if __name__ == "__main__":
    main()