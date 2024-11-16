import yaml
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads the project configuration from a YAML file.
    
    Args:
        config_path (str): The path to the YAML configuration file.
        
    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config