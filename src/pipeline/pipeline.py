from pipeline.l01_raw.l01_pipeline import run_layer_01
from pipeline.l02_intermediate.l02_pipeline import run_layer_02
from pipeline.l03_primary.l03_pipeline import run_layer_03
from pipeline.l04_feature.l04_pipeline import run_layer_04
from pipeline.l05_model_input.l05_pipeline import run_layer_05
from pipeline.l06_models.l06_pipeline import run_layer_06
from pipeline.l07_model_output.l07_pipeline import run_layer_07
from pipeline.l08_reporting.l08_pipeline import run_layer_08

def run_pipeline(config, **connections):
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
