from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from simulation.synthetic_sumo import LANES


FEATURE_COLUMNS = [
    "queue_length",
    "waiting_time",
    "vehicle_count",
    "throughput",
    "signal_phase_index",
    "emergency_count",
    "accident_alert_int",
]


def congestion_level(predicted_queue_length: float) -> str:
    if predicted_queue_length < 30:
        return "low"
    if predicted_queue_length < 120:
        return "medium"
    return "high"


def _with_numeric_features(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    lane_to_index = {lane: index for index, lane in enumerate(LANES)}
    frame["signal_phase_index"] = frame["signal_phase"].map(lane_to_index).fillna(0).astype(int)
    frame["accident_alert_int"] = frame["accident_alert"].astype(int)
    return frame


@dataclass
class TrafficSequenceForecaster:
    window: int = 6
    ridge: float = 0.15
    weights: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None

    def fit(self, metrics: pd.DataFrame) -> "TrafficSequenceForecaster":
        x, y = self.prepare_sequences(metrics)
        if len(x) == 0:
            raise ValueError("Not enough data to train forecaster.")
        self.feature_mean = x.mean(axis=0)
        self.feature_std = x.std(axis=0) + 1e-6
        x_scaled = (x - self.feature_mean) / self.feature_std
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        identity = np.eye(x_design.shape[1])
        identity[0, 0] = 0.0
        self.weights = np.linalg.pinv(x_design.T @ x_design + self.ridge * identity) @ x_design.T @ y
        return self

    def predict_latest(self, metrics: pd.DataFrame) -> pd.DataFrame:
        if self.weights is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Forecaster must be fit before prediction.")
        frame = _with_numeric_features(metrics)
        rows: List[Dict[str, object]] = []
        grouping = ["scenario", "lane_id"] if "scenario" in frame.columns else ["lane_id"]
        for keys, group in frame.groupby(grouping):
            group = group.sort_values("step")
            if len(group) < self.window:
                continue
            latest_window = group.tail(self.window)[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
            prediction = float(self._predict_array(latest_window)[0])
            prediction = max(prediction, 0.0)
            if isinstance(keys, tuple):
                scenario, lane_id = keys
            else:
                scenario, lane_id = "unknown", keys
            rows.append(
                {
                    "scenario": scenario,
                    "lane_id": lane_id,
                    "predicted_queue_length": round(prediction, 3),
                    "congestion_level": congestion_level(prediction),
                }
            )
        return pd.DataFrame(rows)

    def prepare_sequences(self, metrics: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        frame = _with_numeric_features(metrics)
        x_rows: List[np.ndarray] = []
        y_rows: List[float] = []
        grouping = ["scenario", "lane_id"] if "scenario" in frame.columns else ["lane_id"]
        for _, group in frame.groupby(grouping):
            group = group.sort_values("step")
            values = group[FEATURE_COLUMNS].to_numpy(dtype=float)
            targets = group["queue_length"].to_numpy(dtype=float)
            for index in range(self.window, len(values)):
                x_rows.append(values[index - self.window : index].reshape(-1))
                y_rows.append(float(targets[index]))
        if not x_rows:
            return np.empty((0, self.window * len(FEATURE_COLUMNS))), np.empty((0,))
        return np.vstack(x_rows), np.array(y_rows, dtype=float)

    def _predict_array(self, x: np.ndarray) -> np.ndarray:
        assert self.weights is not None
        assert self.feature_mean is not None
        assert self.feature_std is not None
        x_scaled = (x - self.feature_mean) / self.feature_std
        x_design = np.c_[np.ones(len(x_scaled)), x_scaled]
        return x_design @ self.weights


def train_and_predict(metrics: pd.DataFrame, window: int = 6) -> Tuple[TrafficSequenceForecaster, pd.DataFrame]:
    forecaster = TrafficSequenceForecaster(window=window).fit(metrics)
    predictions = forecaster.predict_latest(metrics)
    return forecaster, predictions
