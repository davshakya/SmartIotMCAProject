from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import train_model


def test_load_sample_frame_keeps_feature_columns_only() -> None:
    raw_frame = pd.DataFrame(
        {
            "voltage": [230.0, 231.0],
            "current": [8.0, 8.1],
            "power": [1.8, 1.9],
            "extra": [1, 2],
        }
    )

    with patch.object(train_model.pd, "read_csv", return_value=raw_frame) as read_csv:
        frame = train_model.load_sample_frame()

    read_csv.assert_called_once_with("sample_data.csv")
    assert list(frame.columns) == list(train_model.FEATURE_NAMES)
    assert "extra" not in frame.columns


def test_generate_simulator_normal_frame_has_expected_shape() -> None:
    frame = train_model.generate_simulator_normal_frame(rows=48)

    assert len(frame) == 48
    assert list(frame.columns) == list(train_model.FEATURE_NAMES)
    assert (frame["power"] >= 0.6).all()


def test_load_training_frame_combines_sample_and_simulator_data() -> None:
    sample_frame = pd.DataFrame(
        [{"voltage": 230.0, "current": 8.0, "power": 1.8}],
        columns=train_model.FEATURE_NAMES,
    )
    simulator_frame = pd.DataFrame(
        [
            {"voltage": 231.0, "current": 8.2, "power": 1.9},
            {"voltage": 229.0, "current": 7.8, "power": 1.7},
        ],
        columns=train_model.FEATURE_NAMES,
    )

    with patch.object(train_model, "load_sample_frame", return_value=sample_frame), patch.object(
        train_model, "generate_simulator_normal_frame", return_value=simulator_frame
    ):
        combined, generated = train_model.load_training_frame()

    assert len(combined) == 3
    assert generated.equals(simulator_frame)


def test_build_isolation_forest_returns_fitted_pipeline() -> None:
    values = pd.DataFrame(
        [
            {"voltage": 230.0, "current": 8.0, "power": 1.8},
            {"voltage": 231.0, "current": 8.1, "power": 1.9},
            {"voltage": 232.0, "current": 8.2, "power": 2.0},
            {"voltage": 233.0, "current": 8.3, "power": 2.1},
            {"voltage": 234.0, "current": 8.4, "power": 2.2},
        ],
        columns=train_model.FEATURE_NAMES,
    )

    detector = train_model.build_isolation_forest(values)
    predictions = detector.predict(values.iloc[[0]])

    assert predictions.shape == (1,)


def test_build_lstm_detector_uses_power_column_and_expected_config() -> None:
    values = pd.DataFrame(
        [
            {"voltage": 230.0, "current": 8.0, "power": 1.8},
            {"voltage": 231.0, "current": 8.1, "power": 1.9},
            {"voltage": 232.0, "current": 8.2, "power": 2.0},
        ],
        columns=train_model.FEATURE_NAMES,
    )
    fake_detector = MagicMock()
    fake_detector.fit.return_value = fake_detector

    with patch.object(train_model, "LSTMAnomalyDetector", return_value=fake_detector) as detector_cls:
        result = train_model.build_lstm_detector(values)

    detector_cls.assert_called_once_with(
        feature_count=1,
        sequence_length=8,
        hidden_size=18,
        epochs=60,
        learning_rate=0.003,
        threshold_quantile=99.0,
        random_state=train_model.RANDOM_STATE,
    )
    fit_values = fake_detector.fit.call_args.args[0]
    np.testing.assert_allclose(fit_values.flatten(), values["power"].to_numpy(dtype=float))
    assert result is fake_detector


def test_build_metadata_derives_runtime_thresholds() -> None:
    values = pd.DataFrame(
        [
            {"voltage": 230.0, "current": 8.0, "power": 1.0},
            {"voltage": 231.0, "current": 8.1, "power": 2.0},
            {"voltage": 232.0, "current": 8.2, "power": 3.0},
        ],
        columns=train_model.FEATURE_NAMES,
    )
    fake_lstm = MagicMock(sequence_length=8, threshold_=1.25)

    metadata = train_model.build_metadata(values, fake_lstm)

    assert metadata["features"] == train_model.FEATURE_NAMES
    assert metadata["lstm_sequence_length"] == 8
    assert metadata["lstm_threshold"] == 1.25
    assert metadata["training_rows"] == 3
    assert metadata["primary_runtime_mode"] == "direct"
    assert metadata["power_low_threshold"] < values["power"].min()
    assert metadata["power_high_threshold"] > values["power"].max()


def test_main_persists_all_artifacts() -> None:
    training_frame = pd.DataFrame(
        [
            {"voltage": 230.0, "current": 8.0, "power": 1.8},
            {"voltage": 231.0, "current": 8.1, "power": 1.9},
        ],
        columns=train_model.FEATURE_NAMES,
    )
    lstm_training_frame = pd.DataFrame(
        [{"voltage": 232.0, "current": 8.2, "power": 2.0}],
        columns=train_model.FEATURE_NAMES,
    )
    fake_iforest = object()
    fake_lstm = MagicMock()
    fake_metadata = {"status": "ok"}

    with patch.object(train_model, "ensure_artifact_dir") as ensure_artifact_dir, patch.object(
        train_model, "load_training_frame", return_value=(training_frame, lstm_training_frame)
    ) as load_training_frame, patch.object(
        train_model, "build_isolation_forest", return_value=fake_iforest
    ) as build_isolation_forest, patch.object(
        train_model, "build_lstm_detector", return_value=fake_lstm
    ) as build_lstm_detector, patch.object(
        train_model, "build_metadata", return_value=fake_metadata
    ) as build_metadata, patch.object(
        train_model.joblib, "dump"
    ) as dump_model, patch.object(
        train_model, "save_metadata"
    ) as save_metadata, patch("builtins.print"):
        train_model.main()

    ensure_artifact_dir.assert_called_once_with()
    load_training_frame.assert_called_once_with()
    build_isolation_forest.assert_called_once_with(training_frame)
    build_lstm_detector.assert_called_once_with(lstm_training_frame)
    build_metadata.assert_called_once_with(training_frame, fake_lstm)
    dump_model.assert_called_once_with(fake_iforest, train_model.ISOLATION_FOREST_PATH)
    fake_lstm.save.assert_called_once_with(train_model.LSTM_MODEL_PATH)
    save_metadata.assert_called_once_with(fake_metadata)
