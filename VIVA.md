# Viva Notes

## 1-Minute Introduction

This project is a smart meter monitoring demo that works in direct mode without Kafka or Spark. A simulator generates live meter readings and sends them directly to a FastAPI backend. The backend applies two anomaly detection algorithms: `IsolationForest` for point-wise anomaly detection and an `LSTM` model for power-sequence anomaly detection. The results are displayed on a Flask dashboard, which shows live readings, anomaly decisions, detected events, and model confidence.

## Self Introduction

My project is a smart meter anomaly detection system. It monitors voltage, current, and power readings in real time and identifies abnormal behavior such as theft-like or fault-like events. The project uses Python, FastAPI, Flask, scikit-learn, NumPy, and a custom LSTM implementation.

## Quick Answers

### What is the aim of the project?

The aim is to monitor smart meter readings in real time and detect anomalous power behavior that may indicate electricity theft or meter faults.

### Which machine learning models are used?

This project uses:

- `IsolationForest`
- `LSTM`

### Why did you use Isolation Forest?

Isolation Forest is good for anomaly detection when training data mostly represents normal behavior. It isolates unusual points quickly and works well for single-reading anomalies.

### Why did you use LSTM?

LSTM is useful for sequence data. It checks whether the current power reading fits the recent power pattern from the same meter. This helps detect anomalies that depend on temporal behavior, not just a single point.

### Is this supervised or unsupervised learning?

The anomaly detection part is mainly unsupervised because both detectors are trained on normal readings and then used to identify unusual behavior.

### What are the input features?

The input features are:

- `voltage`
- `current`
- `power`

### What does the model predict?

The backend produces:

- `prediction`: `NORMAL` or `ANOMALY`
- `detected_event`: `NORMAL`, `THEFT`, `FAULT`, or `ANOMALY`

### How is theft detected?

If the reading is anomalous and the power is abnormally low, the system marks it as `THEFT`.

### How is fault detected?

If the reading is anomalous and the power is abnormally high, the system marks it as `FAULT`.

### What is the role of the simulator label?

The simulator label acts as expected output in the demo. It lets us compare the injected event with the backend detection result.

### What is confidence?

Confidence represents how strongly the backend supports its decision based on the anomaly scores from Isolation Forest and LSTM.

### Why is the LSTM sometimes in warmup state?

The LSTM needs a small history of recent readings for each device before it can evaluate sequence anomalies. Until that history is available, only Isolation Forest is active.

### Why did you remove Kafka and Spark?

This version focuses on a simpler direct-mode architecture. The simulator now sends data directly to the backend, which reduces setup complexity and keeps the project easier to run and explain.

## Architecture

```text
meter_simulator.py
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

## Short Viva Answers

### 1. What is anomaly detection?

Anomaly detection means identifying data points or patterns that do not match normal behavior.

### 2. Which ML models did you use?

I used `IsolationForest` and `LSTM`.

### 3. Why two algorithms instead of one?

Isolation Forest checks the current reading directly, while LSTM checks whether the reading fits the recent sequence pattern. Combining both gives stronger detection.

### 4. What dataset did you use?

I used `sample_data.csv` as the normal baseline dataset for training.

### 5. What technologies are used in the project?

I used Python, FastAPI, Flask, scikit-learn, NumPy, requests, and Chart.js.

### 6. What is shown on the dashboard?

The dashboard shows live meter readings, anomaly decisions, detected event types, algorithm status, and confidence values.

### 7. Is this ready for real-world deployment?

No. It is designed for demo and academic use. Real deployment would need more real-world data, evaluation, persistence, and security.
