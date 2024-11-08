import yaml

def load_config(config_path="config/config.yaml"):
    """Loads the project configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config