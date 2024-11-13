from .l01_raw.l01_pipeline import run_layer_01
from .l02_intermediate.l02_pipeline import run_layer_02
from .l03_primary.l03_pipeline import run_layer_03
from .l04_feature.l04_pipeline import run_layer_04
from .l05_model_input.l05_pipeline import run_layer_05
from .l06_models.l06_pipeline import run_layer_06
from .l07_model_output.l07_pipeline import run_layer_07
from .l08_reporting.l08_pipeline import run_layer_08
from src.library.mlflow import start_mlflow_run
from src.library.common import clear_yaml_files

def run_pipeline(config, **connections):
    # Start mlflow run
    start_mlflow_run("main")
    
    # Clear YAML files to start fresh
    clear_yaml_files("./log/")
    
    # Run Layer 1: Raw data ingestion and preprocessing
    l01 = run_layer_01(config, **connections)
    
    # Run Layer 2: Intermediate processing
    l02 = run_layer_02(l01)
    
    # Continue sequentially through each layer, passing outputs to inputs
    l03 = run_layer_03(l02)
    l04 = run_layer_04(l03)
    l05 = run_layer_05(l04)
    model = run_layer_06(l05)
    predictions = run_layer_07(model)
    
    # Final reporting
    run_layer_08(predictions)
