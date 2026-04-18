from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import anomaly_detection as anomaly


TEST_ROOT = Path(__file__).resolve().parent
TEST_RUNTIME_ROOT = TEST_ROOT / ".tmp_runtime"


@pytest.fixture
def workspace_tmp_dir(request: pytest.FixtureRequest):
    TEST_RUNTIME_ROOT.mkdir(exist_ok=True)
    temp_dir = TEST_RUNTIME_ROOT / request.node.name
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir

    shutil.rmtree(temp_dir, ignore_errors=True)
    if TEST_RUNTIME_ROOT.exists() and not any(TEST_RUNTIME_ROOT.iterdir()):
        TEST_RUNTIME_ROOT.rmdir()


def test_sigmoid_and_make_windows() -> None:
    np.testing.assert_allclose(anomaly.sigmoid(np.asarray([0.0])), np.asarray([0.5]))

    values = np.arange(1, 6, dtype=float).reshape(-1, 1)
    sequences, targets = anomaly.make_windows(values, sequence_length=2)

    expected_sequences = np.asarray(
        [
            [[1.0], [2.0]],
            [[2.0], [3.0]],
            [[3.0], [4.0]],
        ]
    )
    expected_targets = np.asarray([[3.0], [4.0], [5.0]])

    np.testing.assert_allclose(sequences, expected_sequences)
    np.testing.assert_allclose(targets, expected_targets)


def test_adam_optimizer_updates_parameters() -> None:
    parameters = {"weight": np.asarray([1.0, -1.0], dtype=float)}
    gradients = {"weight": np.asarray([0.5, -0.25], dtype=float)}
    optimizer = anomaly.AdamOptimizer(parameters, learning_rate=0.1)

    before = parameters["weight"].copy()
    optimizer.step(parameters, gradients)

    assert not np.array_equal(before, parameters["weight"])


def test_simple_lstm_forward_backward_and_fit() -> None:
    model = anomaly.SimpleLSTMRegressor(input_size=1, hidden_size=4, random_state=0)
    sequence = np.asarray([[0.1], [0.2], [0.3]], dtype=float)
    target = np.asarray([0.4], dtype=float)

    prediction, cache = model.forward(sequence)
    gradients = model.backward(target, prediction, cache)

    assert prediction.shape == (1,)
    assert len(cache) == 3
    assert set(gradients) == set(model.parameters)
    for name, gradient in gradients.items():
        assert gradient.shape == model.parameters[name].shape
        assert np.all(np.abs(gradient) <= 1.0 + 1e-9), name

    sequences = np.asarray(
        [
            [[0.1], [0.2], [0.3]],
            [[0.2], [0.3], [0.4]],
            [[0.3], [0.4], [0.5]],
            [[0.4], [0.5], [0.6]],
        ],
        dtype=float,
    )
    targets = np.asarray([[0.4], [0.5], [0.6], [0.7]], dtype=float)
    losses = model.fit(sequences, targets, epochs=2, learning_rate=0.01)

    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)


def test_detector_requires_training_before_use() -> None:
    detector = anomaly.LSTMAnomalyDetector(sequence_length=3, hidden_size=4, epochs=1, random_state=0)

    with pytest.raises(ValueError, match="Detector is not trained"):
        detector._scale(np.asarray([1.0]))
    with pytest.raises(ValueError, match="Detector is not trained"):
        detector.predict(np.asarray([[1.0], [1.1], [1.2]]), np.asarray([1.3]))


def test_detector_fit_rejects_short_series() -> None:
    detector = anomaly.LSTMAnomalyDetector(sequence_length=4, hidden_size=4, epochs=1, random_state=0)

    with pytest.raises(ValueError, match="Not enough samples"):
        detector.fit(np.asarray([1.0, 1.1, 1.2, 1.3]))


def test_detector_fit_handles_zero_std_and_predicts() -> None:
    detector = anomaly.LSTMAnomalyDetector(
        sequence_length=3,
        hidden_size=4,
        epochs=1,
        learning_rate=0.01,
        threshold_quantile=90.0,
        random_state=0,
    )
    constant_values = np.ones(8, dtype=float)

    detector.fit(constant_values)
    score = detector.score(np.asarray([[1.0], [1.0], [1.0]]), np.asarray([1.0]))
    is_anomaly, error = detector.predict(np.asarray([[1.0], [1.0], [1.0]]), np.asarray([1.0]))

    np.testing.assert_allclose(detector.feature_std_, np.asarray([1.0]))
    assert isinstance(score, float)
    assert isinstance(error, float)
    assert isinstance(bool(is_anomaly), bool)


def test_detector_save_and_load_round_trip(workspace_tmp_dir: Path) -> None:
    detector = anomaly.LSTMAnomalyDetector(
        sequence_length=3,
        hidden_size=4,
        epochs=1,
        learning_rate=0.01,
        threshold_quantile=90.0,
        random_state=0,
    )
    values = np.linspace(1.0, 2.0, 8)
    detector.fit(values)

    path = workspace_tmp_dir / "detector_round_trip.npz"

    detector.save(path)
    restored = anomaly.LSTMAnomalyDetector.load(path)

    sample_sequence = values[:3].reshape(-1, 1)
    sample_target = np.asarray([values[3]], dtype=float)

    assert restored.sequence_length == detector.sequence_length
    assert restored.hidden_size == detector.hidden_size
    np.testing.assert_allclose(restored.feature_mean_, detector.feature_mean_)
    np.testing.assert_allclose(restored.feature_std_, detector.feature_std_)
    assert restored.threshold_ == pytest.approx(detector.threshold_)
    assert restored.predict(sample_sequence, sample_target) == detector.predict(sample_sequence, sample_target)


def test_metadata_round_trip_uses_artifact_directory(workspace_tmp_dir: Path) -> None:
    payload = {"power_low_threshold": 0.5, "power_high_threshold": 5.0}

    artifact_dir = workspace_tmp_dir / "artifacts"
    metadata_path = artifact_dir / "metadata.joblib"
    with patch.object(anomaly, "ARTIFACT_DIR", artifact_dir), patch.object(anomaly, "METADATA_PATH", metadata_path):
        anomaly.save_metadata(payload)
        loaded = anomaly.load_metadata()

    assert loaded == payload
