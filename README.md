# Smart Meter Anomaly Detection Dashboard

This project is a direct-mode smart meter monitoring demo. A simulator generates live meter readings, sends them straight to a FastAPI backend, and the backend applies two anomaly detection algorithms:

- `IsolationForest` for point-wise anomaly detection
- a lightweight in-project `LSTM` for sequence anomaly detection

The results are shown on a live Flask dashboard.

## Current Architecture

```text
meter_simulator.py
    |
    v
backend.py (FastAPI + Isolation Forest + LSTM)
    |
    v
flask_app.py
    |
    v
templates/index.html
```

Kafka and Spark are no longer part of the runtime flow.

## Detection Logic

The backend works in two stages:

1. `IsolationForest` checks whether the current reading is unusual compared with normal training data.
2. The `LSTM` looks at the recent `power` sequence from the same device and predicts the next normal power value. If the prediction error is too high, it flags a sequence anomaly.

The backend combines those two signals into:

- `prediction`: `NORMAL` or `ANOMALY`
- `detected_event`: `NORMAL`, `THEFT`, `FAULT`, or `ANOMALY`

`THEFT` and `FAULT` are inferred from the anomaly plus the power level:

- very low power -> `THEFT`
- very high power -> `FAULT`

## Training Data

`sample_data.csv` contains normal smart meter readings and is used as the baseline for both detectors.

Features:

- `voltage`
- `current`
- `power`

Training artifacts are saved under `artifacts/`:

- `isolation_forest.joblib`
- `lstm_detector.npz`
- `detector_metadata.joblib`

## Main Files

### `train_model.py`

Trains both anomaly detectors from `sample_data.csv`.

### `anomaly_detection.py`

Shared anomaly-detection utilities, artifact paths, and the NumPy-based LSTM implementation.

### `backend.py`

Loads the trained detectors, scores incoming readings, keeps recent per-device history for the LSTM, and returns enriched readings to the dashboard.

### `meter_simulator.py`

Generates live meter readings and posts them directly to the backend API.

### `flask_app.py`

Serves the dashboard page and proxies live data from FastAPI.

## Install

```bash
pip install -r requirements.txt
```

Dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `fastapi`
- `uvicorn`
- `flask`
- `requests`

## How to Run

### Recommended

```bash
python run_project.py
```

Optional debug logging:

```bash
python run_project.py --debug true
```

If you already trained the models and want to reuse the saved artifacts:

```bash
python run_project.py --skip-training true
```

### Manual Run Order

1. Train the detectors

```bash
python train_model.py
```

2. Start FastAPI

```bash
python -m uvicorn backend:app --host 0.0.0.0 --port 8000
```

3. Start Flask

```bash
python flask_app.py
```

4. Start the simulator

```bash
python meter_simulator.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Dashboard Fields

Each reading shown in the dashboard includes:

- simulator `label`
- combined anomaly `prediction`
- inferred `detected_event`
- `confidence`
- individual algorithm results for `Isolation Forest` and `LSTM`

## Notes

- The LSTM needs a short warm-up history for each device before it can score sequences.
- `THEFT` and `FAULT` detection are demo-oriented heuristics built on top of anomaly detection.
- The project is intended for academic demonstration, not production utility deployment.

## Viva Notes

Viva preparation is available in [VIVA.md](/d:/SmartIotMCAProject/VIVA.md:1).
