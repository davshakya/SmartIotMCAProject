from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend


class FakeIsolationForest:
    def __init__(self, score: float, is_anomaly: bool) -> None:
        self.score = score
        self.is_anomaly = is_anomaly
        self.frames: list = []

    def decision_function(self, frame):
        self.frames.append(frame.copy())
        return np.asarray([self.score], dtype=float)

    def predict(self, frame):
        self.frames.append(frame.copy())
        return np.asarray([-1 if self.is_anomaly else 1], dtype=int)


class FakeLSTMDetector:
    def __init__(self, sequence_length: int = 3, response: tuple[bool, float] = (False, 0.0)) -> None:
        self.sequence_length = sequence_length
        self.response = response
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def predict(self, sequence: np.ndarray, target: np.ndarray) -> tuple[bool, float]:
        self.calls.append((np.asarray(sequence, dtype=float).copy(), np.asarray(target, dtype=float).copy()))
        return self.response


@pytest.fixture(autouse=True)
def restore_backend_state():
    original_data = backend.DATA
    original_history = backend.DEVICE_HISTORY
    original_iforest = backend.isolation_forest
    original_lstm = backend.lstm_detector
    original_metadata = backend.metadata

    backend.DATA = []
    backend.metadata = {
        "power_low_threshold": 0.5,
        "power_high_threshold": 5.0,
        "lstm_threshold": 1.0,
    }
    backend.DEVICE_HISTORY = defaultdict(lambda: deque(maxlen=3))

    yield

    backend.DATA = original_data
    backend.DEVICE_HISTORY = original_history
    backend.isolation_forest = original_iforest
    backend.lstm_detector = original_lstm
    backend.metadata = original_metadata


@pytest.fixture
def client():
    with TestClient(backend.app) as test_client:
        yield test_client


def install_models(
    *,
    iforest_score: float,
    iforest_is_anomaly: bool,
    lstm_response: tuple[bool, float] = (False, 0.0),
    sequence_length: int = 3,
    metadata: dict | None = None,
) -> FakeLSTMDetector:
    backend.isolation_forest = FakeIsolationForest(iforest_score, iforest_is_anomaly)
    backend.lstm_detector = FakeLSTMDetector(sequence_length=sequence_length, response=lstm_response)
    backend.metadata = {
        "power_low_threshold": 0.5,
        "power_high_threshold": 5.0,
        "lstm_threshold": 1.0,
        **(metadata or {}),
    }
    backend.DEVICE_HISTORY = defaultdict(lambda: deque(maxlen=sequence_length))
    backend.DATA = []
    return backend.lstm_detector


def make_item(
    *,
    power: float,
    voltage: float = 230.0,
    current: float = 8.0,
    device_id: str | None = "meter-1",
    label: str | None = None,
) -> dict:
    item = {
        "voltage": voltage,
        "current": current,
        "power": power,
    }
    if device_id is not None:
        item["device_id"] = device_id
    if label is not None:
        item["label"] = label
    return item


def prime_history(device_id: str, powers: list[float]) -> None:
    history = backend.DEVICE_HISTORY[device_id]
    for power in powers:
        history.append(np.asarray([230.0, 8.0, power], dtype=float))


def test_feature_builders_and_helper_logic() -> None:
    item = make_item(power=1.8)
    features = backend.build_feature_vector(item)
    frame = backend.build_feature_frame(features)

    np.testing.assert_allclose(features, np.asarray([230.0, 8.0, 1.8], dtype=float))
    assert list(frame.columns) == ["voltage", "current", "power"]
    assert backend.isolation_forest_confidence(0.0) == pytest.approx(0.5)
    assert backend.lstm_confidence(5.0, 0.0) == 0.0
    assert backend.lstm_confidence(3.0, 1.0) == pytest.approx(0.9502, abs=1e-4)
    assert backend.classify_event(0.2, True) == "THEFT"
    assert backend.classify_event(6.0, True) == "FAULT"
    assert backend.classify_event(2.0, True) == "ANOMALY"
    assert backend.classify_event(2.0, False) == "NORMAL"
    assert (
        backend.combine_confidence(
            prediction="ANOMALY",
            iforest_is_anomaly=False,
            iforest_conf=0.1,
            lstm_supports_alert=False,
            lstm_conf=None,
        )
        == 0.5
    )
    assert (
        backend.combine_confidence(
            prediction="NORMAL",
            iforest_is_anomaly=False,
            iforest_conf=0.2,
            lstm_supports_alert=False,
            lstm_conf=0.4,
        )
        == 0.7
    )


def test_analyze_item_returns_normal_during_lstm_warmup() -> None:
    install_models(iforest_score=0.6, iforest_is_anomaly=False)

    result = backend.analyze_item(make_item(power=1.8, device_id=None, label="NORMAL"))

    assert result["prediction"] == "NORMAL"
    assert result["detected_event"] == "NORMAL"
    assert result["anomaly_source"] == "None"
    assert result["match"] is True
    assert result["algorithms"]["lstm"]["status"] == "warmup"
    assert result["algorithms"]["lstm"]["history_available"] == 1
    assert "METER" in backend.DEVICE_HISTORY


def test_analyze_item_flags_theft_from_isolation_forest() -> None:
    install_models(iforest_score=-0.4, iforest_is_anomaly=True)

    result = backend.analyze_item(make_item(power=0.2, label="THEFT"))

    assert result["prediction"] == "ANOMALY"
    assert result["detected_event"] == "THEFT"
    assert result["anomaly_source"] == "Isolation Forest"
    assert result["confidence"] == pytest.approx(round(backend.isolation_forest_confidence(-0.4), 4), abs=1e-4)


def test_analyze_item_ignores_low_confidence_lstm_anomaly() -> None:
    lstm_detector = install_models(
        iforest_score=0.9,
        iforest_is_anomaly=False,
        lstm_response=(True, 0.2),
        sequence_length=3,
    )
    prime_history("meter-1", [1.0, 1.1, 1.2])

    result = backend.analyze_item(make_item(power=1.3, label="NORMAL"))

    assert result["prediction"] == "NORMAL"
    assert result["detected_event"] == "NORMAL"
    assert result["anomaly_source"] == "None"
    assert result["algorithms"]["lstm"]["status"] == "ready"
    assert result["algorithms"]["lstm"]["is_anomaly"] is True
    assert len(lstm_detector.calls) == 1
    sequence, target = lstm_detector.calls[0]
    np.testing.assert_allclose(sequence[:, 0], np.asarray([1.0, 1.1, 1.2]))
    np.testing.assert_allclose(target, np.asarray([1.3]))


def test_analyze_item_raises_alert_from_high_confidence_lstm_signal() -> None:
    install_models(
        iforest_score=0.8,
        iforest_is_anomaly=False,
        lstm_response=(True, 3.0),
        sequence_length=3,
    )
    prime_history("meter-1", [1.0, 1.1, 1.2])

    result = backend.analyze_item(make_item(power=3.0, label="ANOMALY"))

    assert result["prediction"] == "ANOMALY"
    assert result["detected_event"] == "ANOMALY"
    assert result["anomaly_source"] == "LSTM"
    assert result["match"] is True
    assert result["confidence"] >= backend.LSTM_ALERT_CONFIDENCE


def test_analyze_item_combines_both_models_for_fault_event() -> None:
    install_models(
        iforest_score=-0.2,
        iforest_is_anomaly=True,
        lstm_response=(True, 3.0),
        sequence_length=3,
    )
    prime_history("meter-1", [4.8, 5.0, 5.1])

    result = backend.analyze_item(make_item(power=6.2, label="FAULT"))

    assert result["prediction"] == "ANOMALY"
    assert result["detected_event"] == "FAULT"
    assert result["anomaly_source"] == "Isolation Forest, LSTM"
    expected = max(backend.isolation_forest_confidence(-0.2), backend.lstm_confidence(3.0, 1.0))
    assert result["confidence"] == pytest.approx(round(expected, 4), abs=1e-4)


def test_device_history_is_kept_separate_per_meter() -> None:
    lstm_detector = install_models(
        iforest_score=0.6,
        iforest_is_anomaly=False,
        lstm_response=(True, 3.0),
        sequence_length=2,
    )

    backend.analyze_item(make_item(power=1.0, device_id="meter-a"))
    backend.analyze_item(make_item(power=1.0, device_id="meter-b"))
    backend.analyze_item(make_item(power=1.1, device_id="meter-a"))
    backend.analyze_item(make_item(power=1.2, device_id="meter-a"))

    assert len(lstm_detector.calls) == 1
    sequence, _ = lstm_detector.calls[0]
    np.testing.assert_allclose(sequence[:, 0], np.asarray([1.0, 1.1]))
    assert len(backend.DEVICE_HISTORY["meter-b"]) == 1


def test_api_routes_store_recent_results_only(client, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_analyze_item(item: dict) -> dict:
        return {
            **item,
            "prediction": "NORMAL",
            "detected_event": "NORMAL",
            "confidence": 0.75,
            "match": True,
            "anomaly_source": "None",
            "algorithms": {},
        }

    monkeypatch.setattr(backend, "analyze_item", fake_analyze_item)

    for index in range(backend.MAX_VISIBLE_READINGS + 5):
        response = client.post("/data", json={"index": index})
        assert response.status_code == 200
        assert response.json() == {"ok": True, "prediction": "NORMAL"}

    get_response = client.get("/data")

    assert get_response.status_code == 200
    assert len(get_response.json()) == backend.MAX_VISIBLE_READINGS
    assert get_response.json()[0]["index"] == 5
