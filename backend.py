from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from anomaly_detection import (
    FEATURE_NAMES,
    ISOLATION_FOREST_PATH,
    LSTM_MODEL_PATH,
    METADATA_PATH,
    LSTMAnomalyDetector,
    load_metadata,
)


MAX_VISIBLE_READINGS = 60
LSTM_ALERT_CONFIDENCE = 0.85

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_artifact(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"Missing model artifact: {path}. Run `python train_model.py` first.")
    return path


metadata = load_metadata() if METADATA_PATH.exists() else {}
isolation_forest = joblib.load(require_artifact(ISOLATION_FOREST_PATH))
lstm_detector = LSTMAnomalyDetector.load(require_artifact(LSTM_MODEL_PATH))
sequence_length = lstm_detector.sequence_length

DATA: list[dict] = []
DEVICE_HISTORY: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=sequence_length))


def build_feature_vector(item: dict) -> np.ndarray:
    return np.asarray([float(item[name]) for name in FEATURE_NAMES], dtype=float)


def build_feature_frame(features: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([features], columns=FEATURE_NAMES)


def isolation_forest_confidence(score: float) -> float:
    return float(1.0 / (1.0 + np.exp(5.0 * score)))


def lstm_confidence(error: float, threshold: float) -> float:
    if threshold <= 0.0:
        return 0.0
    ratio = error / threshold
    return float(max(0.0, min(0.999, 1.0 - np.exp(-ratio))))


def classify_event(power: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "NORMAL"
    if power <= float(metadata.get("power_low_threshold", 0.5)):
        return "THEFT"
    if power >= float(metadata.get("power_high_threshold", 5.0)):
        return "FAULT"
    return "ANOMALY"


def combine_confidence(
    prediction: str,
    iforest_is_anomaly: bool,
    iforest_conf: float,
    lstm_supports_alert: bool,
    lstm_conf: float | None,
) -> float:
    if prediction == "ANOMALY":
        signals = []
        if iforest_is_anomaly:
            signals.append(iforest_conf)
        if lstm_supports_alert:
            signals.append(float(lstm_conf or 0.0))
        return round(max(signals) if signals else 0.5, 4)

    normal_signals = [1.0 - iforest_conf]
    if lstm_conf is not None:
        normal_signals.append(1.0 - lstm_conf)
    return round(float(sum(normal_signals) / len(normal_signals)), 4)


def analyze_item(item: dict) -> dict:
    device_id = str(item.get("device_id") or "METER")
    features = build_feature_vector(item)
    power = float(features[2])
    feature_frame = build_feature_frame(features)

    iforest_score = float(isolation_forest.decision_function(feature_frame)[0])
    iforest_is_anomaly = bool(isolation_forest.predict(feature_frame)[0] == -1)
    iforest_conf = isolation_forest_confidence(iforest_score)

    history = DEVICE_HISTORY[device_id]
    lstm_ready = len(history) >= lstm_detector.sequence_length
    lstm_is_anomaly: bool | None = None
    lstm_error: float | None = None
    lstm_conf: float | None = None

    if lstm_ready:
        sequence = np.asarray([[entry[2]] for entry in history], dtype=float)
        lstm_target = np.asarray([power], dtype=float)
        lstm_is_anomaly, lstm_error = lstm_detector.predict(sequence, lstm_target)
        lstm_conf = lstm_confidence(lstm_error, float(metadata.get("lstm_threshold", 0.0)))

    lstm_drives_alert = bool(lstm_is_anomaly) and float(lstm_conf or 0.0) >= LSTM_ALERT_CONFIDENCE
    combined_anomaly = iforest_is_anomaly or lstm_drives_alert
    prediction = "ANOMALY" if combined_anomaly else "NORMAL"
    detected_event = classify_event(power, combined_anomaly)
    confidence = combine_confidence(
        prediction=prediction,
        iforest_is_anomaly=iforest_is_anomaly,
        iforest_conf=iforest_conf,
        lstm_supports_alert=lstm_drives_alert,
        lstm_conf=lstm_conf,
    )

    anomaly_sources = []
    if iforest_is_anomaly:
        anomaly_sources.append("Isolation Forest")
    if lstm_drives_alert:
        anomaly_sources.append("LSTM")

    history.append(features)

    enriched_item = dict(item)
    enriched_item["prediction"] = prediction
    enriched_item["detected_event"] = detected_event
    enriched_item["confidence"] = confidence
    enriched_item["match"] = enriched_item.get("label") == detected_event
    enriched_item["anomaly_source"] = ", ".join(anomaly_sources) if anomaly_sources else "None"
    enriched_item["algorithms"] = {
        "isolation_forest": {
            "is_anomaly": iforest_is_anomaly,
            "score": round(iforest_score, 4),
            "confidence": round(iforest_conf, 4),
        },
        "lstm": {
            "status": "ready" if lstm_ready else "warmup",
            "is_anomaly": lstm_is_anomaly,
            "score": None if lstm_error is None else round(lstm_error, 4),
            "confidence": None if lstm_conf is None else round(lstm_conf, 4),
            "required_history": lstm_detector.sequence_length,
            "history_available": len(history),
        },
    }
    return enriched_item


@app.post("/data")
def receive(item: dict):
    enriched_item = analyze_item(item)
    DATA.append(enriched_item)
    if len(DATA) > MAX_VISIBLE_READINGS:
        del DATA[:-MAX_VISIBLE_READINGS]
    return {"ok": True, "prediction": enriched_item["prediction"]}


@app.get("/data")
def get_data():
    return DATA
