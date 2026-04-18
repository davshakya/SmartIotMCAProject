from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


FEATURE_NAMES = ("voltage", "current", "power")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ISOLATION_FOREST_PATH = ARTIFACT_DIR / "isolation_forest.joblib"
LSTM_MODEL_PATH = ARTIFACT_DIR / "lstm_detector.npz"
METADATA_PATH = ARTIFACT_DIR / "detector_metadata.joblib"


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def make_windows(data: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    targets = []
    for index in range(len(data) - sequence_length):
        sequences.append(data[index : index + sequence_length])
        targets.append(data[index + sequence_length])
    return np.asarray(sequences), np.asarray(targets)


class AdamOptimizer:
    def __init__(self, parameters: dict[str, np.ndarray], learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.timestep = 0
        self.moment_1 = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.moment_2 = {name: np.zeros_like(value) for name, value in parameters.items()}

    def step(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]) -> None:
        self.timestep += 1
        for name, gradient in gradients.items():
            self.moment_1[name] = self.beta1 * self.moment_1[name] + (1.0 - self.beta1) * gradient
            self.moment_2[name] = self.beta2 * self.moment_2[name] + (1.0 - self.beta2) * (gradient**2)

            moment_1_hat = self.moment_1[name] / (1.0 - self.beta1**self.timestep)
            moment_2_hat = self.moment_2[name] / (1.0 - self.beta2**self.timestep)
            parameters[name] -= self.learning_rate * moment_1_hat / (np.sqrt(moment_2_hat) + self.epsilon)


class SimpleLSTMRegressor:
    def __init__(self, input_size: int, hidden_size: int = 16, random_state: int = 42) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.default_rng(random_state)
        weight_scale = np.sqrt(2.0 / (input_size + hidden_size))

        self.parameters = {
            "Wf": rng.normal(0.0, weight_scale, size=(input_size, hidden_size)),
            "Uf": rng.normal(0.0, weight_scale, size=(hidden_size, hidden_size)),
            "bf": np.ones(hidden_size),
            "Wi": rng.normal(0.0, weight_scale, size=(input_size, hidden_size)),
            "Ui": rng.normal(0.0, weight_scale, size=(hidden_size, hidden_size)),
            "bi": np.zeros(hidden_size),
            "Wg": rng.normal(0.0, weight_scale, size=(input_size, hidden_size)),
            "Ug": rng.normal(0.0, weight_scale, size=(hidden_size, hidden_size)),
            "bg": np.zeros(hidden_size),
            "Wo": rng.normal(0.0, weight_scale, size=(input_size, hidden_size)),
            "Uo": rng.normal(0.0, weight_scale, size=(hidden_size, hidden_size)),
            "bo": np.zeros(hidden_size),
            "Wy": rng.normal(0.0, weight_scale, size=(hidden_size, input_size)),
            "by": np.zeros(input_size),
        }

    def forward(self, sequence: np.ndarray) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
        hidden_state = np.zeros(self.hidden_size)
        cell_state = np.zeros(self.hidden_size)
        cache: list[dict[str, np.ndarray]] = []

        for x_t in sequence:
            forget_gate = sigmoid(
                x_t @ self.parameters["Wf"] + hidden_state @ self.parameters["Uf"] + self.parameters["bf"]
            )
            input_gate = sigmoid(
                x_t @ self.parameters["Wi"] + hidden_state @ self.parameters["Ui"] + self.parameters["bi"]
            )
            candidate = np.tanh(
                x_t @ self.parameters["Wg"] + hidden_state @ self.parameters["Ug"] + self.parameters["bg"]
            )
            output_gate = sigmoid(
                x_t @ self.parameters["Wo"] + hidden_state @ self.parameters["Uo"] + self.parameters["bo"]
            )

            previous_hidden = hidden_state
            previous_cell = cell_state
            cell_state = forget_gate * cell_state + input_gate * candidate
            hidden_state = output_gate * np.tanh(cell_state)

            cache.append(
                {
                    "x": x_t,
                    "h_prev": previous_hidden,
                    "c_prev": previous_cell,
                    "f": forget_gate,
                    "i": input_gate,
                    "g": candidate,
                    "o": output_gate,
                    "c": cell_state,
                    "h": hidden_state,
                }
            )

        prediction = hidden_state @ self.parameters["Wy"] + self.parameters["by"]
        return prediction, cache

    def backward(
        self, target: np.ndarray, prediction: np.ndarray, cache: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        prediction_gradient = 2.0 * (prediction - target) / target.size

        gradients["Wy"] += np.outer(cache[-1]["h"], prediction_gradient)
        gradients["by"] += prediction_gradient

        hidden_gradient = prediction_gradient @ self.parameters["Wy"].T
        cell_gradient = np.zeros(self.hidden_size)

        for step in reversed(cache):
            tanh_cell = np.tanh(step["c"])
            output_gradient = hidden_gradient * tanh_cell
            output_pre_activation = output_gradient * step["o"] * (1.0 - step["o"])

            cell_gradient = cell_gradient + hidden_gradient * step["o"] * (1.0 - tanh_cell**2)
            forget_gradient = cell_gradient * step["c_prev"]
            forget_pre_activation = forget_gradient * step["f"] * (1.0 - step["f"])

            input_gradient = cell_gradient * step["g"]
            input_pre_activation = input_gradient * step["i"] * (1.0 - step["i"])

            candidate_gradient = cell_gradient * step["i"]
            candidate_pre_activation = candidate_gradient * (1.0 - step["g"] ** 2)

            gradients["Wf"] += np.outer(step["x"], forget_pre_activation)
            gradients["Uf"] += np.outer(step["h_prev"], forget_pre_activation)
            gradients["bf"] += forget_pre_activation

            gradients["Wi"] += np.outer(step["x"], input_pre_activation)
            gradients["Ui"] += np.outer(step["h_prev"], input_pre_activation)
            gradients["bi"] += input_pre_activation

            gradients["Wg"] += np.outer(step["x"], candidate_pre_activation)
            gradients["Ug"] += np.outer(step["h_prev"], candidate_pre_activation)
            gradients["bg"] += candidate_pre_activation

            gradients["Wo"] += np.outer(step["x"], output_pre_activation)
            gradients["Uo"] += np.outer(step["h_prev"], output_pre_activation)
            gradients["bo"] += output_pre_activation

            hidden_gradient = (
                forget_pre_activation @ self.parameters["Uf"].T
                + input_pre_activation @ self.parameters["Ui"].T
                + candidate_pre_activation @ self.parameters["Ug"].T
                + output_pre_activation @ self.parameters["Uo"].T
            )
            cell_gradient = cell_gradient * step["f"]

        for gradient in gradients.values():
            np.clip(gradient, -1.0, 1.0, out=gradient)
        return gradients

    def fit(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 80,
        learning_rate: float = 0.003,
    ) -> list[float]:
        optimizer = AdamOptimizer(self.parameters, learning_rate=learning_rate)
        rng = np.random.default_rng(42)
        losses: list[float] = []

        for _ in range(epochs):
            epoch_losses = []
            for index in rng.permutation(len(sequences)):
                prediction, cache = self.forward(sequences[index])
                loss = float(np.mean((prediction - targets[index]) ** 2))
                gradients = self.backward(targets[index], prediction, cache)
                optimizer.step(self.parameters, gradients)
                epoch_losses.append(loss)
            losses.append(float(np.mean(epoch_losses)))
        return losses

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        prediction, _ = self.forward(sequence)
        return prediction


@dataclass
class LSTMAnomalyDetector:
    feature_count: int = 1
    sequence_length: int = 8
    hidden_size: int = 16
    epochs: int = 80
    learning_rate: float = 0.003
    threshold_quantile: float = 97.5
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = SimpleLSTMRegressor(
            input_size=self.feature_count,
            hidden_size=self.hidden_size,
            random_state=self.random_state,
        )
        self.feature_mean_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None
        self.threshold_: float | None = None
        self.training_loss_: list[float] = []

    def fit(self, values: np.ndarray) -> "LSTMAnomalyDetector":
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if len(values) <= self.sequence_length:
            raise ValueError("Not enough samples to train the LSTM detector.")

        self.feature_mean_ = values.mean(axis=0)
        self.feature_std_ = values.std(axis=0)
        self.feature_std_[self.feature_std_ == 0.0] = 1.0

        scaled = self._scale(values)
        sequences, targets = make_windows(scaled, self.sequence_length)
        self.training_loss_ = self.model.fit(
            sequences,
            targets,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )

        errors = np.asarray([self._prediction_error(sequence, target) for sequence, target in zip(sequences, targets)])
        self.threshold_ = float(np.percentile(errors, self.threshold_quantile))
        return self

    def _scale(self, values: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise ValueError("Detector is not trained.")
        values = np.asarray(values, dtype=float)
        if values.ndim == 1 and self.feature_count == 1:
            values = values.reshape(-1, 1) if values.size > 1 else values.reshape(1)
        return (values - self.feature_mean_) / self.feature_std_

    def _prediction_error(self, sequence: np.ndarray, target: np.ndarray) -> float:
        predicted = self.model.predict(sequence)
        return float(np.mean((predicted - target) ** 2))

    def score(self, sequence: np.ndarray, target: np.ndarray) -> float:
        scaled_sequence = self._scale(sequence)
        scaled_target = self._scale(target)
        if self.feature_count == 1:
            scaled_sequence = np.asarray(scaled_sequence, dtype=float).reshape(self.sequence_length, 1)
            scaled_target = np.asarray(scaled_target, dtype=float).reshape(1)
        return self._prediction_error(scaled_sequence, scaled_target)

    def predict(self, sequence: np.ndarray, target: np.ndarray) -> tuple[bool, float]:
        if self.threshold_ is None:
            raise ValueError("Detector is not trained.")
        error = self.score(sequence, target)
        return error > self.threshold_, error

    def save(self, path: Path | str = LSTM_MODEL_PATH) -> None:
        if self.feature_mean_ is None or self.feature_std_ is None or self.threshold_ is None:
            raise ValueError("Detector is not trained.")
        path = Path(path)
        np.savez(
            path,
            feature_count=self.feature_count,
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            threshold_quantile=self.threshold_quantile,
            random_state=self.random_state,
            feature_mean=self.feature_mean_,
            feature_std=self.feature_std_,
            threshold=self.threshold_,
            training_loss=np.asarray(self.training_loss_, dtype=float),
            **self.model.parameters,
        )

    @classmethod
    def load(cls, path: Path | str = LSTM_MODEL_PATH) -> "LSTMAnomalyDetector":
        payload = np.load(Path(path))
        detector = cls(
            feature_count=int(payload["feature_count"]),
            sequence_length=int(payload["sequence_length"]),
            hidden_size=int(payload["hidden_size"]),
            epochs=int(payload["epochs"]),
            learning_rate=float(payload["learning_rate"]),
            threshold_quantile=float(payload["threshold_quantile"]),
            random_state=int(payload["random_state"]),
        )
        detector.feature_mean_ = np.asarray(payload["feature_mean"], dtype=float)
        detector.feature_std_ = np.asarray(payload["feature_std"], dtype=float)
        detector.threshold_ = float(payload["threshold"])
        detector.training_loss_ = payload["training_loss"].astype(float).tolist()
        for name in detector.model.parameters:
            detector.model.parameters[name] = np.asarray(payload[name], dtype=float)
        return detector


def save_metadata(metadata: dict[str, float | int | str | tuple[str, ...]]) -> None:
    ensure_artifact_dir()
    joblib.dump(metadata, METADATA_PATH)


def load_metadata() -> dict[str, float | int | str | tuple[str, ...]]:
    return joblib.load(METADATA_PATH)
