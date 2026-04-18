# Smart Meter IoT Monitoring with Theft and Fault Detection

This project is a real-time smart meter monitoring demo that simulates meter readings, streams them through Kafka and Spark, classifies each reading using machine learning, and shows the results on a live web dashboard.

The system predicts one of three classes:

- `NORMAL`
- `THEFT`
- `FAULT`

It also supports a direct fallback mode so the dashboard and ML prediction can still work even if Spark is unavailable.

## Project Overview

The project demonstrates how IoT-style energy readings can be:

- generated from multiple smart meters
- streamed through Kafka
- processed by Spark Structured Streaming
- classified by an ML model
- displayed live in a browser dashboard

The main use case is academic demonstration of:

- smart meter monitoring
- theft and fault detection
- streaming pipelines
- ML-based classification
- dashboard visualization

## Current Architecture

### Full streaming pipeline

```text
meter_simulator.py
    |
    v
Kafka topic: smart-meter
    |
    v
spark_stream.py
    |
    v
backend.py (FastAPI + ML model)
    |
    v
flask_app.py
    |
    v
templates/index.html
```

### Direct fallback pipeline

If Spark is missing or fails, the launcher switches to direct mode automatically:

```text
meter_simulator.py --mode direct
    |
    v
backend.py
    |
    v
flask_app.py
    |
    v
dashboard
```

In both modes, the ML model is still applied in the FastAPI backend.

## Main Components

### `run_project.py`

This is the main launcher for the project.

It:

- starts Kafka in KRaft mode if needed
- trains the model before startup
- starts FastAPI
- starts Spark using the bundled PySpark `spark-submit`
- adds the Kafka connector package automatically
- falls back to direct mode if Spark cannot run
- starts the simulator
- starts the Flask dashboard

### `train_model.py`

This script trains the ML classifier and saves it to `model.pkl`.

Current model:

- `RandomForestClassifier`

Features used:

- `voltage`
- `current`
- `power`

Predicted classes:

- `NORMAL`
- `THEFT`
- `FAULT`

### `backend.py`

This is the FastAPI inference backend.

It:

- loads `model.pkl`
- receives meter readings
- predicts `NORMAL`, `THEFT`, or `FAULT`
- attaches confidence score
- keeps recent readings in memory
- returns recent data to the dashboard

Endpoints:

- `POST /data`
- `GET /data`

### `meter_simulator.py`

This script generates live readings from multiple simulated smart meters.

It:

- produces readings for multiple devices
- simulates daily usage changes
- injects theft-like and fault-like values
- supports Kafka mode and direct mode

### `spark_stream.py`

This is the Spark Structured Streaming job.

It:

- consumes Kafka topic `smart-meter`
- parses JSON meter records
- forwards records to the FastAPI backend

### `flask_app.py`

This serves the dashboard HTML page and fetches processed data from FastAPI.

### `templates/index.html`

This is the dashboard UI.

It shows:

- live power graph
- prediction counters
- simulator label counters
- recent event cards
- prediction confidence
- play/pause control for the graph
- color-coded theft and fault points

### `sample_data.csv`

Base dataset used as the normal class source for training.

### `model.pkl`

Serialized trained classifier created by `train_model.py`.

## Machine Learning Design

The current project uses supervised multiclass classification.

### Model used

- `RandomForestClassifier`

### Input features

- `voltage`
- `current`
- `power`

### Output classes

- `NORMAL`
- `THEFT`
- `FAULT`

### How training works

`train_model.py` takes `sample_data.csv` as the `NORMAL` class.

Then it creates synthetic labeled data for the other two classes:

- `THEFT`
  - based on very low power values
  - roughly in the range `0.0` to `0.5`

- `FAULT`
  - based on very high power values
  - roughly in the range `20.0` to `30.0`

These ranges match the simulator behavior, so the classifier learns to recognize the same patterns it later sees during runtime.

### Runtime prediction flow

For every new reading:

1. the backend extracts `voltage`, `current`, and `power`
2. the model predicts one class
3. the backend adds:
   - `prediction`
   - `detected_event`
   - `confidence`
4. the dashboard shows the result live

### Important limitation

This project is still a prototype.

The `THEFT` and `FAULT` classes are generated synthetically from normal data, so the model is suitable for demos and academic presentation, but not yet for real utility deployment.

## Theft and Fault Logic

### Theft basis

Theft is represented by abnormally low power values.

In this project:

- simulator theft samples use power around `0.0` to `0.5`
- the model learns that this pattern maps to `THEFT`

### Fault basis

Fault is represented by abnormally high power values.

In this project:

- simulator fault samples use power around `20.0` to `30.0`
- the model learns that this pattern maps to `FAULT`

### Normal basis

Normal readings stay in the usual power range found in `sample_data.csv`, which is far lower than the fault range and far higher than the theft range.

## Example Data Format

Incoming simulator record:

```json
{
  "device_id": "METER_1",
  "timestamp": "2026-04-18 08:00:00",
  "voltage": 231.44,
  "current": 10.28,
  "power": 2.38,
  "label": "NORMAL"
}
```

Backend-enriched record:

```json
{
  "device_id": "METER_1",
  "timestamp": "2026-04-18 08:00:00",
  "voltage": 231.44,
  "current": 10.28,
  "power": 2.38,
  "label": "NORMAL",
  "prediction": "NORMAL",
  "detected_event": "NORMAL",
  "confidence": 0.98
}
```

## Requirements

You should have:

- Python 3.9 or newer
- Java
- Kafka in KRaft mode
- internet access for first-time Spark Kafka package download

Optional but recommended:

- a virtual environment

## Python Dependencies

Install the required packages:

```bash
pip install pandas scikit-learn joblib kafka-python pyspark requests fastapi uvicorn flask numpy
```

## How to Run

### Recommended one-command run

```bash
python3 run_project.py
```

If you are using a virtual environment:

```bash
python run_project.py
```

### Debug run

To save full logs:

```bash
python run_project.py --debug true
```

Accepted debug truthy values include:

- `true`
- `True`
- `1`
- `yes`
- `on`

### What the launcher does

The launcher:

- checks Kafka on `localhost:9092`
- starts local Kafka in KRaft mode if needed
- creates the `smart-meter` topic if needed
- retrains the model
- starts FastAPI on port `8000`
- starts Spark using the PySpark bundled `spark-submit`
- injects the Kafka connector package:
  - `org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1`
- starts the simulator
- starts Flask on port `5000`

Then open:

```text
http://127.0.0.1:5000
```

## Manual Run Order

### 1. Train the model

```bash
python train_model.py
```

### 2. Start FastAPI backend

```bash
python -m uvicorn backend:app --host 0.0.0.0 --port 8000
```

### 3. Start Flask dashboard

```bash
python flask_app.py
```

### 4. Start simulator

Kafka mode:

```bash
python meter_simulator.py
```

Direct mode:

```bash
python meter_simulator.py --mode direct
```

### 5. Start Spark streaming manually

If you want to run Spark yourself, use the PySpark bundled `spark-submit` with the Kafka package:

```bash
SPARK_HOME=/home/devendra/myenv/lib/python3.12/site-packages/pyspark \
/home/devendra/myenv/lib/python3.12/site-packages/pyspark/bin/spark-submit \
--packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 \
spark_stream.py
```

Adjust the Python version path if your environment uses a different version.

## Dashboard Explanation

The dashboard includes:

- summary cards
- a power timeline graph
- recent event cards
- model predictions
- simulator labels
- prediction confidence
- play/pause graph control

### How to read the graph

- X-axis: timestamp
- Y-axis: power
- blue line: actual power values
- red points: predicted `THEFT`
- purple points: predicted `FAULT`
- black points: predicted `NORMAL`

If the graph shows very low power and red theft points, the model is detecting theft-like behavior.
If the graph shows very high power and purple points, the model is detecting fault-like behavior.

### Meaning of fields

- `Label`: simulator-provided expected class
- `Prediction`: model output
- `Detected`: same as prediction in current version
- `Confidence`: model confidence score

### Match count

`Prediction Match Count` shows how many recent readings have the same simulator label and ML prediction.

## Logging

### Normal mode

Normal runs print useful status only and do not save full logs.

### Debug mode

When you pass `--debug true`, a full log is saved under:

```text
logs/run_project_YYYYMMDD_HHMMSS.log
```

Debug logs include:

- launcher steps
- subprocess output
- Kafka output
- Spark output
- FastAPI output
- simulator output
- fallback decisions

## Common Issues

### Spark fails immediately

Typical causes:

- incorrect Spark path
- missing Kafka connector
- wrong Spark/Scala package version

Current project fix:

- launcher uses bundled PySpark `spark-submit`
- launcher adds the matching Kafka connector package automatically

### Dashboard does not open

Possible causes:

- Flask is not running
- backend is not running
- project stopped after an earlier failure
- port `5000` is in use

Check:

- terminal status messages
- `logs/` if running in debug mode

### Kafka is not found

The launcher expects Kafka under:

```text
~/kafka
```

Expected files:

- `~/kafka/bin/kafka-storage.sh`
- `~/kafka/bin/kafka-server-start.sh`
- `~/kafka/bin/kafka-topics.sh`
- `~/kafka/config/kraft/server.properties`

## Current Limitations

- backend stores only recent records in memory
- restarting backend clears dashboard history
- theft and fault training data are synthetic
- project is intended for demo and academic use
- not suitable yet for production smart-grid deployment

## Suggested Future Improvements

- use real labeled utility meter data
- add persistent storage
- compute proper evaluation metrics
- add per-device filters
- export reports
- containerize with Docker Compose
- move settings into environment variables
- add authentication and access control

## Viva Notes

All viva preparation material has been moved to [VIVA.md](/home/devendra/SmartIotMCAProject/VIVA.md:1).

That file contains:

- viva question and answer material
- 1-minute introduction
- self introduction
- quick revision notes
- short viva answers

## Project Summary

This project is a working smart meter monitoring prototype with:

- live multi-device simulation
- Kafka + Spark streaming pipeline
- direct fallback mode
- ML classification into `NORMAL`, `THEFT`, and `FAULT`
- live dashboard visualization
- configurable debug logging

It is designed as a strong academic demo that clearly shows IoT simulation, data streaming, machine learning classification, and dashboard monitoring in one integrated system.
