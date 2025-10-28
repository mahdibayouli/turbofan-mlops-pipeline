# MLOps for Predictive Maintenance (NASA Turbofan)

This is an end-to-end MLOps pipeline for predictive maintenance. The system analyzes real-time sensor data from NASA's C-MAPSS turbofan engine dataset to predict failures (anomaly detection) and diagnose causes (RAG).

## Local Setup

1.  Clone the repository:
    ```bash
    git clone git@github.com:mahdibayouli/turbofan-mlops-pipeline.git
    cd turbofan-mlops-pipeline
    ```

2.  **Download and Prepare Data:**
    * Download the full dataset ZIP file from: **[NASA C-MAPSS Data](https://data.nasa.gov/docs/legacy/CMAPSSData.zip)**
    * Extract the contents.
    * Place the resulting `CMAPSSData` folder into your local repository's `data/` directory.
    * **The final structure must be:** e.g. `data/CMAPSSData/train_FD001.txt`

3.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows: .\venv\Scripts\activate
    ```

4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---
*(This README is a WIP)*