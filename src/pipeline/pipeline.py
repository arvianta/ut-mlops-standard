from pipeline.l01_raw.l01_pipeline import run_layer_01
from pipeline.l02_intermediate.l02_pipeline import run_layer_02
from pipeline.l03_primary.l03_pipeline import run_layer_03
from pipeline.l04_feature.l04_pipeline import run_layer_04
from pipeline.l05_model_input.l05_pipeline import run_layer_05
from pipeline.l06_models.l06_pipeline import run_layer_06
from pipeline.l07_model_output.l07_pipeline import run_layer_07
from pipeline.l08_reporting.l08_pipeline import run_layer_08

def run_pipeline(config, databricks_client=None, gcs_client=None, bq_client=None):
    # Run Layer 1: Raw data ingestion and preprocessing
    data_layer_1 = run_layer_01()
    
    # Run Layer 2: Intermediate processing
    data_layer_2 = run_layer_02(data_layer_1)
    
    # Continue sequentially through each layer, passing outputs to inputs
    data_layer_3 = run_layer_03(data_layer_2)
    data_layer_4 = run_layer_04(data_layer_3)
    data_layer_5 = run_layer_05(data_layer_4)
    model = run_layer_06(data_layer_5)
    predictions = run_layer_07(model)
    
    # Final reporting
    run_layer_08(predictions)
