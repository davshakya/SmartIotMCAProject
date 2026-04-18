# Project Milestone Reports

Project Title: Smart Meter Anomaly Detection Dashboard

This milestone report describes how the project was built from historical smart meter data through preprocessing, anomaly detection, live monitoring, dashboard visualization, alert logic, testing, evaluation, and documentation.

## Milestone 1: Collect Historical Smart Meter Data

The first milestone focused on preparing the baseline data required for model training. Historical smart meter readings were organized in `sample_data.csv` and used as the reference dataset for normal behavior.

Work completed:

- historical smart meter readings prepared in `sample_data.csv`
- baseline normal behavior defined for training
- core numerical features identified for downstream processing

Outcome:

The project gained a stable historical dataset for anomaly-detection training.

## Milestone 2: Preprocess Time-Series Dataset

The second milestone prepared the data for AI processing. The relevant features were selected, normal patterns were structured, and the time-series data was shaped for both point-wise and sequential anomaly detection.

Work completed:

- selected `voltage`, `current`, and `power` as the main features
- prepared a clean baseline dataset for anomaly learning
- applied scaling for `IsolationForest`
- created fixed-length windows for `LSTM` sequence training

Outcome:

The raw readings were converted into model-ready time-series input.

## Milestone 3: Implement Anomaly Detection Algorithms

This milestone introduced the core AI logic of the project. Two anomaly detection models were implemented:

- `IsolationForest` for point anomaly detection
- `LSTM` for sequence anomaly detection

Work completed:

- `IsolationForest` training implemented in `train_model.py`
- custom `LSTM` anomaly detector implemented in `anomaly_detection.py`
- model artifacts saved under `artifacts/`
- inference logic integrated into `backend.py`

Outcome:

The system can now identify both individual abnormal readings and abnormal sequential behavior.

## Milestone 4: Deploy Monitoring Agent Flow

This milestone introduced the live monitoring flow. The simulator was used as a smart meter agent that sends real-time readings directly to the backend.

Work completed:

- simulator implemented for multiple meter devices
- direct API-based telemetry pipeline created
- per-device history handling added for `LSTM`
- launcher simplified to direct-mode runtime

Outcome:

The project now supports continuous monitoring of live smart meter readings.

## Milestone 5: Visualize Anomalies Using Dashboard

The dashboard milestone focused on observability. The system was given a live interface that displays readings, detections, and model behavior.

Work completed:

- Flask dashboard implemented
- live power timeline added
- recent event cards added
- anomaly counters and model outputs displayed

Outcome:

Users can observe anomaly detection results clearly during live operation.

## Milestone 6: Configure Automated Alert Triggers

This milestone converted model decisions into alert-ready outputs.

Work completed:

- anomaly trigger rules added in the backend
- confidence output added
- event classification into `NORMAL`, `THEFT`, `FAULT`, or `ANOMALY`
- anomaly source tracking added

Outcome:

The system now produces alert-style signals whenever suspicious meter behavior is detected.

## Milestone 7: Test Detection Accuracy Using Simulated Spikes

The simulator was extended to inject abnormal low and high readings so detection behavior could be tested repeatedly.

Work completed:

- theft-like low-power spikes simulated
- fault-like high-power spikes simulated
- labels attached to simulated events
- backend decisions compared with expected event patterns

Outcome:

The project can be tested under controlled abnormal scenarios without needing real utility field faults.

## Milestone 8: Evaluate False Positives And Precision Behavior

This milestone focused on detection quality. The backend and dashboard expose match information that helps evaluate false positives and detection usefulness.

Work completed:

- `match` tracking added to backend results
- dashboard summary includes match-related information
- false-positive behavior can be observed during normal runs
- anomaly-vs-label comparisons support precision-style evaluation

Outcome:

The project supports practical quality evaluation for live anomaly detection behavior.

## Milestone 9: Deployment And System Integration

This milestone focused on putting the components together into one working system.

Work completed:

- backend service integrated with trained models
- simulator integrated with backend API
- dashboard integrated with backend data
- one-command launcher created in `run_project.py`

Outcome:

The full Smart Meter Anomaly Detection Dashboard now runs as one connected application.

## Milestone 10: Documentation And Viva Preparation

The final milestone documented the entire project flow for submission and presentation.

Work completed:

- full project explanation written in `README.md`
- viva notes prepared in `VIVA.md`
- milestone summary aligned with implemented functionality

Outcome:

The project is now documented as a complete smart meter monitoring and anomaly detection system.

## Final Status Summary

- Collect historical smart meter data: Completed
- Preprocess time-series dataset: Completed
- Implement anomaly detection algorithms: Completed
- Deploy monitoring agent flow: Completed
- Visualize anomalies using dashboard: Completed
- Configure automated alert triggers: Completed
- Test detection accuracy using simulated spikes: Completed
- Evaluate false positives and precision behavior: Completed
- Deploy and integrate system components: Completed
- Document project workflow and integration: Completed
