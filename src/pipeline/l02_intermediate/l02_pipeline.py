from .node_standard_processing import standard_processing
from .node_utils import suggest_attributes_for_processing
from src.library.common import load_files_to_dict

def run_layer_02(raw_paths):
    data = load_files_to_dict(raw_paths)
    attr = suggest_attributes_for_processing(data)
    processed_data = standard_processing(data, attr)
    
    return processed_data