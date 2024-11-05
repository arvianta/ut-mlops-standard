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
│   ├── 1_raw/                  # Holds raw data as ingested from external sources
│   ├── 2_intermediate/         # Stores data after general cleansing and standard transformations
│   ├── 3_primary/              # Contains cleaned, linked, and normalized data
│   ├── 4_feature/              # Stores engineered features for modeling
│   ├── 5_model_input/          # Contains data ready for model input after selection and split
│   ├── 6_models/               # Holds model configurations and model binaries
│   ├── 7_model_output/         # Stores results of model predictions and performance metrics
│   ├── 8_reporting/            # Contains generated reports and final analysis outputs
├── src/
│   ├── lib/                    # Common functions library for data cleansing, modeling, etc.
│   ├── connections/            # Scripts for establishing connections to data sources
│   ├── ingestion/              # Scripts for data ingestion from various sources
│   ├── 1_raw_layer/            # Scripts for loading and handling raw data
│   ├── 2_intermediate_layer/   # Data cleansing scripts (e.g., null handling, type conversion)
│   ├── 3_primary_layer/        # Data linking, normalization, and joining scripts
│   ├── 4_feature_layer/        # Scripts for feature engineering
│   ├── 5_model_input_layer/    # Scripts for data selection, imputation, and handling outliers
│   ├── 6_models_layer/         # Model selection, training, and tuning scripts
│   ├── 7_model_output_layer/   # Scripts to store and manage model results
│   ├── 8_reporting_layer/      # Scripts to generate reports for model evaluation
│   └── main.py                 # Main entry point to initiate the pipeline
├── notebooks/                  # Jupyter notebooks for exploration, experimentation, and EDA
├── config/                     # Configuration files for connections to GCP, Databricks, etc.
└── README.md                   # Project overview and structure explanation
```

## Directory Overview
### 1. ``data/``
This directory contains structured subfolders to handle data as it progresses through the project lifecycle. Data is organized in a sequence of layers that mirror the data pipeline.
- ``1_raw/``: Contains raw, unprocessed data as ingested.
- ``2_intermediate/``: Data after initial cleansing (e.g., null handling, type standardization).
- ``3_primary/``: Cleaned and joined data with primary/foreign keys, consistent naming conventions, and normalization.
- ``4_feature/``: Feature-engineered data ready for use in modeling.
- ``5_model_input/``: Finalized dataset for modeling, including selected features and imputed values.
- ``6_models/``: Stores trained models and configurations.
- ``7_model_output/``: Contains model predictions and evaluation metrics.
- ``8_reporting/``: Holds final reports generated from model analysis and performance metrics.

### 2. ``src/``
This directory houses the main codebase, organized by layers, with each layer handling specific processing steps. Scripts in each layer make use of the functions in the ``lib`` directory for standardized, reusable code.
- ``lib/``: Contains common functions for data cleansing, feature engineering, modeling, and MLflow logging.
- ``connections/``: Scripts for establishing connections to external data sources (GCP, Databricks, etc.).
- ``ingestion/``: Scripts for data ingestion from various sources, including GCS, Databricks Unity Catalog, and local files.
- ``1_raw_layer/``: Scripts for loading and processing raw data.
- ``2_intermediate_layer/``: Cleansing and standardizing data types and formats.
- ``3_primary_layer/``: Joins and normalizes data, preparing a structured dataset.
- ``4_feature_layer/``: Applies feature engineering to enhance predictive power.
- ``5_model_input_layer/``: Conducts feature selection, outlier handling, and splits data.
- ``6_models_layer/``: Manages model training, hyperparameter tuning, and selection.
- ``7_model_output_layer/``: Stores and organizes model output and evaluation results.
- ``8_reporting_layer/``: Generates reports and case-specific analyses.
- ``main.py``: The main entry point that initializes and runs the pipeline.

### 3. ``notebooks/``
This directory contains Jupyter notebooks for exploratory data analysis (EDA), experimentation, and data science exploration. Notebooks are named to follow the layer sequence (e.g., ``1_raw_eda.ipynb``, ``2_intermediate_cleaning.ipynb``) to maintain an intuitive flow.

### 4. ``config/``
Holds configuration files and connection information for cloud services (e.g., Google Cloud, Databricks). This enables seamless integration with data sources and deployment environments.

## Usage Notes
- **MLflow Logging**: Logging is centralized within ``src/lib/mlflow_logging.py``. Each layer imports this module as needed to standardize tracking and ensure traceability across all processes.
- **Library**: Reusable functions in ``lib`` make it easy to apply standardized operations across all layers. Add to this library as needed to support data cleansing, modeling, or additional functionality.