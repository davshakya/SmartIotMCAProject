# Smart Meter Anomaly Detection Dashboard

This project is a smart meter monitoring and anomaly-detection system. It collects historical meter readings, preprocesses the time-series data, applies `IsolationForest` and `LSTM` anomaly detection, streams live readings through a direct API pipeline, and visualizes results on a dashboard.

The system is designed as an academic demo of how AI can be integrated into smart energy monitoring for theft-like and fault-like behavior detection.

## Project Goals

This project is built around the following goals:

- collect historical smart meter readings
- preprocess time-series datasets
- implement anomaly detection using `IsolationForest` and `LSTM`
- deploy monitoring agents that send live readings
- visualize anomalies on dashboards
- configure automated alert triggers
- test detection behavior using simulated spikes
- evaluate false positives and precision-related behavior
- support deployment through a backend + dashboard architecture
- document the AI workflow and monitoring integration

## Current Repository Scope

The repository currently implements the full local demo version of the project:

- historical baseline data in `sample_data.csv`
- anomaly model training in [train_model.py](/d:/SmartIotMCAProject/train_model.py:1)
- custom LSTM utilities in [anomaly_detection.py](/d:/SmartIotMCAProject/anomaly_detection.py:1)
- FastAPI inference service in [backend.py](/d:/SmartIotMCAProject/backend.py:1)
- live meter simulator in [meter_simulator.py](/d:/SmartIotMCAProject/meter_simulator.py:1)
- Flask dashboard in [flask_app.py](/d:/SmartIotMCAProject/flask_app.py:1)
- dashboard UI in [templates/index.html](/d:/SmartIotMCAProject/templates/index.html:1)
- one-command launcher in [run_project.py](/d:/SmartIotMCAProject/run_project.py:1)

This version runs in direct mode without Kafka or Spark.

## Project Architecture

```text
sample_data.csv
    |
    v
train_model.py
    |
    v
artifacts/
    |
    v
meter_simulator.py --> backend.py --> flask_app.py --> templates/index.html
```

### Flow Summary

1. historical smart meter readings are prepared in `sample_data.csv`
2. `train_model.py` preprocesses the data and trains anomaly detectors
3. model artifacts are saved under `artifacts/`
4. `meter_simulator.py` generates live readings for multiple meters
5. `backend.py` scores each reading with `IsolationForest` and `LSTM`
6. `flask_app.py` serves dashboard data
7. the dashboard visualizes anomalies, events, and model outputs

## Historical Data Collection

The historical dataset is stored in `sample_data.csv`.

It provides the normal baseline behavior used for model training. The dataset contains the following numerical features:

- `voltage`
- `current`
- `power`

These readings represent typical smart meter operation and are used as the reference profile for anomaly detection.

## Time-Series Preprocessing

The preprocessing workflow is handled in `train_model.py`.

It includes:

- selecting the three core features
- combining historical and simulator-like normal patterns
- preparing a clean baseline for anomaly learning
- scaling features for `IsolationForest`
- building fixed-length windows from the `power` series for `LSTM`
- learning an anomaly threshold from normal sequence prediction error

This gives the project both point-wise and sequential context during runtime.

## Anomaly Detection Models

The project uses two complementary algorithms:

### `IsolationForest`

- detects unusual individual readings
- works well for outlier-style anomalies
- helps catch sudden abnormal values immediately

### `LSTM`

- analyzes recent sequential behavior
- predicts the next expected `power` value
- flags a sequence anomaly when prediction error becomes too high

## Runtime Decision Logic

The backend combines both model signals into:

- `prediction`: `NORMAL` or `ANOMALY`
- `detected_event`: `NORMAL`, `THEFT`, `FAULT`, or `ANOMALY`
- `confidence`
- `anomaly_source`

In this project:

- very low abnormal power is interpreted as `THEFT`
- very high abnormal power is interpreted as `FAULT`

This makes the dashboard easier to explain during demos and viva presentations.

## Monitoring Agent

`meter_simulator.py` acts as the live monitoring agent in this repository.

It:

- generates readings for multiple meter devices
- stamps each reading with device ID and timestamp
- simulates normal usage variation
- injects abnormal low and high spikes
- sends data directly to the FastAPI backend

This mirrors how a real smart meter agent could forward live readings from distributed devices to a central monitoring service.

## Dashboard And Visualization

The dashboard is served through Flask and rendered with Chart.js.

It shows:

- total visible readings
- combined anomalies
- detected theft-like events
- detected fault-like events
- `IsolationForest` alerts
- `LSTM` alerts
- latest decision
- live power timeline
- recent event cards
- confidence values
- algorithm status and anomaly source

This helps users visually inspect how the detection pipeline behaves in real time.

## Automated Alert Triggers

The project includes alert-driving logic inside the backend.

An alert condition is triggered when:

- `IsolationForest` marks the current reading as anomalous, or
- `LSTM` reports a strong enough sequence anomaly to support an alert

The alert state is exposed through:

- `prediction`
- `detected_event`
- `confidence`
- `anomaly_source`

In the current repository, alerting is visible in the dashboard and API response. In a future deployed version, the same alert state can be forwarded to external channels such as SMS, email, or utility monitoring systems.

## Simulated Spike Testing

The simulator injects controlled abnormal values to test the detection behavior.

Current abnormal test cases include:

- low power spikes representing theft-like behavior
- high power spikes representing fault-like behavior

Because the simulator also attaches a label, the backend output can be compared against the expected event type during testing.

## Evaluation Approach

The project supports practical evaluation through simulator labels and backend decisions.

Important evaluation indicators include:

- `match count`
- false positives
- anomaly frequency on normal behavior
- precision-style comparison between injected events and detected events

The backend already returns a `match` field, and the dashboard displays match-oriented summaries. This makes it easier to assess whether the detectors are behaving sensibly during live runs.

## Model Artifacts

Training saves the following artifacts under `artifacts/`:

- `isolation_forest.joblib`
- `lstm_detector.npz`
- `detector_metadata.joblib`

Generate them with:

```bash
python train_model.py
```

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `fastapi`
- `uvicorn`
- `flask`
- `requests`

## How To Run

### Recommended

```bash
python run_project.py
```

### Debug mode

```bash
python run_project.py --debug true
```

### Reuse existing model artifacts

```bash
python run_project.py --skip-training true
```

### Manual startup order

1. Train the models

```bash
python train_model.py
```

2. Start the backend

```bash
python -m uvicorn backend:app --host 0.0.0.0 --port 8000
```

3. Start the dashboard

```bash
python flask_app.py
```

4. Start the live simulator

```bash
python meter_simulator.py
```

Then open:

```text
http://127.0.0.1:5000
```

## API Behavior

### `POST /data`

Receives a smart meter reading and returns a processed anomaly decision.

### `GET /data`

Returns recent processed readings for the dashboard.

Each enriched reading can include:

- device data
- simulator label
- `prediction`
- `detected_event`
- `confidence`
- `anomaly_source`
- per-algorithm details for `IsolationForest` and `LSTM`

## Deployment Direction

The current repository is a local academic demo, but its structure supports straightforward deployment:

- host the FastAPI backend on a server
- host the dashboard service with the backend or behind a reverse proxy
- deploy monitoring agents where smart meter data is produced
- forward alert states to external notification systems
- optionally store long-term readings in a database

## AI Workflow And Monitoring Integration

The AI workflow in this project is:

1. collect historical smart meter data
2. preprocess the time-series baseline
3. train anomaly detection models
4. receive live readings from monitoring agents
5. run inference continuously in the backend
6. visualize events in the dashboard
7. trigger alerts for suspicious behavior
8. evaluate false positives and match quality
9. retrain and tune thresholds if needed

This is the main integration idea of the project: the AI models are part of the full monitoring loop, not just offline experiments.

## Limitations

This project is still an academic prototype.

Current limitations:

- the data is simulated and demo-oriented
- alert delivery is shown through API/dashboard state rather than external messaging integrations
- long-term persistent storage is not implemented
- evaluation is observable in the live app but not exported into a dedicated report generator
- the system is designed for demonstration, not production utility deployment

## Related Documents

- project milestones: [MILESTONE_REPORTS.md](/d:/SmartIotMCAProject/MILESTONE_REPORTS.md:1)
- viva preparation: [VIVA.md](/d:/SmartIotMCAProject/VIVA.md:1)
