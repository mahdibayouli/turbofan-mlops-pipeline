# MLOps for Predictive Maintenance (NASA Turbofan)

This is an end-to-end MLOps pipeline for predictive maintenance. The system analyzes time series sensor data from NASA's C-MAPSS turbofan engine dataset to predict failures (anomaly detection) and diagnose causes (RAG).

## Local Setup

1. Clone the repository:

    ```bash
    git clone git@github.com:mahdibayouli/turbofan-mlops-pipeline.git
    cd turbofan-mlops-pipeline
    ```

2. Download and Prepare Data
    * Download the full dataset ZIP file from: **[NASA C-MAPSS Data](https://data.nasa.gov/docs/legacy/CMAPSSData.zip)**
    * Extract the contents.
    * Place the resulting `CMAPSSData` folder into your local repository's `data/` directory.
    * This example illustrates how fhe final structure must look like: `data/CMAPSSData/train_FD001.txt`

3. Create and activate a Python virtual environment:

    ```bash
    # Linux/macOS:
    python3 -m venv venv
    source venv/bin/activate

    # Windows (PowerShell):
    python -m venv venv
    .\venv\Scripts\activate
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Current Workflow (v0)

At this stage of the project, the pipeline has three main steps:

1. **Explore the dataset (optional, but recommended)**
   Open the first notebook: `notebooks\01_data_exploration.ipynb`
   This notebook:
    * loads `data/CMAPSSData/train_FD001.txt`
    * assigns column names from the official `data/CMAPSSData/readme.txt`
    * checks basic statistics and missing values
    * visualizes run to failure behavior for FD001
2. **Run preprocessing to create artifacts**
   Fit the scaler and generate preprocessing metadata:

   ```bash
   python -m src.preprocess
   ```

   This will create and save the following files under `models/`:
   * `scaler.joblib`: fitted [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) on the useful sensor columns
   * `artifacts_info.json`: which sensor columns to drop and which to scale

    These artifacts are later reused by the training script, the evaluation notebook, and the FastAPI API.
3. **Train the anomaly detector (LSTM autoencoder)**
   Train on healthy FD001 windows:

   ```bash
   python -m src.train
   ```

   This script trains a two-layer LSTM Seq2Seq autoencoder to reconstruct fixed size windows of healthy data and saves the trained weights as `models/detector.pth`.

   The detailed design choices for the architecture will be documented later in a dedicated `docs/model.md`

---

(This README file is still a WIP)