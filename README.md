# UT Data Science Project 

## Project Overview
Welcome to **UT Data Science Project**! This project aims to [standardization]. The structure and standardization aim to simplify collaboration and maintenance, following MLOps best practices.

---

## Getting Started
1. **Clone the repository:**
```bash
git clone https://github.com/user/reponame
cd project_name
```
2. **Choose the data ingestion method**
3. **Build and run the pipeline**

## Project Structure

The following is an overview of the project structure, with each directory explained in detail.

```plaintext
project/
├── data/
│   ├── 1_raw/                   # Raw data as ingested from external sources
│   ├── 2_intermediate/          # Data after general cleansing and standard transformations
│   ├── 3_primary/               # Cleaned, linked, and normalized data
│   ├── 4_feature/               # Engineered features for modeling
│   ├── 5_model_input/           # Data ready for model input
│   ├── 6_model/                 # Model binaries and configurations
│   ├── 7_model_output/          # Model predictions and performance metrics
│   ├── 8_reporting/             # Generated reports and final analysis outputs
│   └── tmp/                     # Temporary data storage
├── src/
│   ├── main.py                  # Main entry point for the pipeline
│   ├── __init__.py
│   ├── library/                 # Common functions for data processing, MLflow integration, etc.
│   ├── pipeline/                # Pipeline processing steps organized by layer
│   │   ├── l00_preraw/          # Preprocessing raw data
│   │   ├── l01_raw/             # Raw data ingestion and processing
│   │   ├── l02_intermediate/    # Cleansing and standard processing
│   │   ├── l03_primary/         # Data linking and normalization
│   │   ├── l04_feature/         # Feature engineering scripts
│   │   ├── l05_model_input/     # Data preparation for model input
│   │   ├── l06_models/          # Model training, tuning, and storage
│   │   ├── l07_model_output/    # Model predictions and evaluations
│   │   └── l08_reporting/       # Report generation
│   └── utils/                   # Utility scripts and connectors for external data sources
│       ├── config.py            # Configuration file handling
│       └── connectors/          # Databricks, GCP connectors for data ingestion
├── notebooks/                   # Jupyter notebooks for EDA and data exploration
├── config/                      # Configuration files for project settings
├── log/                         # Log files for tracking
├── .gitignore                   # Files to ignore in Git
├── README.md                    # Project overview and structure explanation
└── requirements.txt             # Project dependencies
```

## Directory Overview
### 1. ``data/``
This directory contains structured subfolders to manage data across the project lifecycle. Each folder represents a processing stage:
- ``1_raw/``: Contains raw, unprocessed data as ingested.
- ``2_intermediate/``: Data after initial cleansing.
- ``3_primary/``: Cleaned and joined data.
- ``4_feature/``: Feature-engineered data for modeling.
- ``5_model_input/``: Finalized datasets for model training.
- ``6_models/``: Stored model binaries and configurations.
- ``7_model_output/``: Model predictions and evaluation metrics.
- ``8_reporting/``: Reports from analysis and performance metrics.
- ``tmp/``: Temporary storage for files.

### 2. ``src/``
Contains the main codebase, organized by layer, with each layer handling a different pipeline stage. Reusable functions are housed in ``library/``
- ``main.py``: The main entry point that initializes and runs the pipeline.
- ``library/``: Contains common functions for data processing and MLflow integration.
- ``pipeline/``: Scripts organized by processing layers (e.g., raw, feature engineering, model training).
    - ``l00_preraw/``: Ensuring data is in a proper tabular format. (optional)
    - ``l01_raw/``: Scripts for loading and processing raw data.
    - ``l02_intermediate/``: Cleansing and standard processing.
    - ``l03_primary/``: Data linking and normalization.
    - ``l04_feature/``: Feature engineering for model training.
    - ``l05_model_input/``: Data preparation for model input.
    - ``l06_models/``: Model training, tuning, and storage.
    - ``l07_model_output/``: Model predictions and evaluations.
    - ``l08_reporting/``: Report generation and analysis.
- ``utils/``: Utility scripts and connectors.
    - ``config.py``: Handles project configurations.
    - ``connectors/``: Establishes connections to external sources (e.g., Databricks, GCP).

### 3. ``notebooks/``
This directory contains Jupyter notebooks for exploratory data analysis (EDA), experimentation, and data science exploration. Notebooks are named to follow the layer sequence (e.g., ``1_raw_eda.ipynb``, ``2_intermediate_cleaning.ipynb``) to maintain an intuitive flow.

### 4. ``config/``
Contains configuration files (``config.yaml``) for specifying ingestion methods, cloud connections, and project settings.

## Usage Notes
- **MLflow Logging**: Logging is centralized within ``src/library/mlflow.py``. Each layer imports this module as needed to standardize tracking and ensure traceability across all processes.
- **Configuration**: Modify settings in ``config/config.yaml`` to specify ingestion methods and data connections.