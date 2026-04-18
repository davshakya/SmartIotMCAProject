from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anomaly_detection import (
    FEATURE_NAMES,
    ISOLATION_FOREST_PATH,
    LSTM_MODEL_PATH,
    LSTMAnomalyDetector,
    ensure_artifact_dir,
    save_metadata,
)


RANDOM_STATE = 42
ISOLATION_FOREST_CONTAMINATION = 0.04
SIMULATED_NORMAL_ROWS = 480


def load_sample_frame() -> pd.DataFrame:
    frame = pd.read_csv("sample_data.csv")
    return frame[list(FEATURE_NAMES)].copy()


def generate_simulator_normal_frame(rows: int = SIMULATED_NORMAL_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    samples = []
    for index in range(rows):
        hour = index % 24
        voltage = rng.uniform(210.0, 240.0)
        current = rng.uniform(5.0, 15.0)
        power = voltage * current / 1000.0
        if 18 <= hour <= 23:
            power *= 1.5
        elif 0 <= hour <= 6:
            power *= 0.55
        power += rng.normal(0.0, 0.08)
        power = max(0.6, power)
        samples.append(
            {
                "voltage": round(float(voltage), 4),
                "current": round(float(current), 4),
                "power": round(float(power), 4),
            }
        )
    return pd.DataFrame(samples, columns=FEATURE_NAMES)


def load_training_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_frame = load_sample_frame()
    simulator_frame = generate_simulator_normal_frame()
    combined_frame = pd.concat([sample_frame, simulator_frame], ignore_index=True)
    return combined_frame, simulator_frame


def build_isolation_forest(values: pd.DataFrame) -> Pipeline:
    detector = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "isolation_forest",
                IsolationForest(
                    contamination=ISOLATION_FOREST_CONTAMINATION,
                    n_estimators=250,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    detector.fit(values)
    return detector


def build_lstm_detector(values: pd.DataFrame) -> LSTMAnomalyDetector:
    detector = LSTMAnomalyDetector(
        feature_count=1,
        sequence_length=8,
        hidden_size=18,
        epochs=60,
        learning_rate=0.003,
        threshold_quantile=99.0,
        random_state=RANDOM_STATE,
    )
    detector.fit(values[["power"]].to_numpy(dtype=float))
    return detector


def build_metadata(values: pd.DataFrame, lstm_detector: LSTMAnomalyDetector) -> dict[str, float | int | str | tuple[str, ...]]:
    power_min = float(values["power"].min())
    power_max = float(values["power"].max())
    power_std = float(values["power"].std())

    return {
        "features": FEATURE_NAMES,
        "power_low_threshold": max(0.1, power_min - (0.35 * power_std)),
        "power_high_threshold": power_max + (0.75 * power_std),
        "lstm_sequence_length": lstm_detector.sequence_length,
        "lstm_threshold": float(lstm_detector.threshold_ or 0.0),
        "lstm_feature_name": "power",
        "isolation_forest_contamination": ISOLATION_FOREST_CONTAMINATION,
        "training_rows": int(len(values)),
        "primary_runtime_mode": "direct",
    }


def main() -> None:
    ensure_artifact_dir()
    training_frame, lstm_training_frame = load_training_frame()
    isolation_forest = build_isolation_forest(training_frame)
    lstm_detector = build_lstm_detector(lstm_training_frame)
    metadata = build_metadata(training_frame, lstm_detector)

    joblib.dump(isolation_forest, ISOLATION_FOREST_PATH)
    lstm_detector.save(LSTM_MODEL_PATH)
    save_metadata(metadata)

    print("Isolation Forest and LSTM anomaly detectors trained successfully.", flush=True)
    print(f"Saved Isolation Forest artifact to {ISOLATION_FOREST_PATH}", flush=True)
    print(f"Saved LSTM artifact to {LSTM_MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
