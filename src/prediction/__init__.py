"""Congestion prediction modules."""

try:
    from .lstm_forecaster import TrafficSequenceForecaster, train_and_predict
except ImportError:
    from lstm_forecaster import TrafficSequenceForecaster, train_and_predict

__all__ = ["TrafficSequenceForecaster", "train_and_predict"]
