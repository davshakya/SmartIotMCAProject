# Project Milestone Reports

Project Title: Smart Meter Anomaly Detection Dashboard

## 1. Collect Historical Data

Baseline smart meter readings were collected and stored in `sample_data.csv`. These readings provide the normal operating profile used for anomaly detection. A simulator was also prepared to generate live readings with `device_id`, `timestamp`, `voltage`, `current`, and `power`.

Deliverables completed:
- baseline dataset prepared in `sample_data.csv`
- multi-device simulator created in `meter_simulator.py`
- core electrical features standardized for downstream processing

## 2. Preprocess Dataset

The dataset was prepared for anomaly detection by selecting the key numerical features: `voltage`, `current`, and `power`. Since the new design focuses on learning normal behavior, the training flow uses normal readings as the reference baseline.

Deliverables completed:
- feature selection finalized as `voltage`, `current`, and `power`
- normal-behavior dataset prepared for detector training
- training pipeline organized in `train_model.py`

## 3. Implement Algorithms

The core ML milestone introduced two anomaly detection algorithms:

- `IsolationForest` for point-wise anomaly detection
- `LSTM` for sequence-based anomaly detection

Both models are trained from `sample_data.csv`. Isolation Forest detects unusual individual readings, while the LSTM checks whether the current reading fits recent device behavior.

Deliverables completed:
- `IsolationForest` integrated for direct anomaly scoring
- NumPy-based `LSTM` implemented for sequence anomaly detection
- shared artifact logic added in `anomaly_detection.py`
- training artifacts saved under `artifacts/`

## 4. Deploy Monitoring Agents

The runtime pipeline was simplified to direct mode. The simulator now sends readings directly to the FastAPI backend, which performs anomaly detection and stores enriched results for the dashboard.

Deliverables completed:
- direct API pipeline implemented
- FastAPI backend updated for dual-detector inference
- per-device recent history added for LSTM scoring
- launcher simplified in `run_project.py`

## 5. Create Dashboard

The dashboard was updated to visualize anomaly detection results instead of the old multiclass classifier. It now shows combined anomalies, theft-like and fault-like events, Isolation Forest alerts, LSTM alerts, and recent event details.

Deliverables completed:
- dashboard updated for anomaly detection terminology
- live graph retained for power monitoring
- event cards expanded with algorithm-level details
- summary cards aligned with the new backend response

## 6. Test and Document

The project was documented for the direct-only architecture, and the viva notes were updated to explain the new algorithms and runtime flow.

Deliverables completed:
- documentation refreshed in `README.md`
- viva notes updated in `VIVA.md`
- milestone report aligned with the new design

## Short Status Summary

- Collect Historical Data: Completed
- Preprocess Dataset: Completed
- Implement Algorithms: Completed
- Deploy Monitoring Agents: Completed
- Create Dashboard: Completed
- Test and Document: Completed
