from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


VEHICLE_COLUMNS = [
    "timestamp",
    "lane_id",
    "car_count",
    "bus_count",
    "truck_count",
    "motorcycle_count",
    "emergency_count",
]


@dataclass
class VehicleCounter:
    """Counts vehicles from video when dependencies exist, else documents fallback."""

    model_name: str = "yolov8n.pt"

    @property
    def yolo_available(self) -> bool:
        return bool(importlib.util.find_spec("ultralytics"))

    @property
    def opencv_available(self) -> bool:
        return bool(importlib.util.find_spec("cv2"))

    def count_video(self, video_path: str | Path) -> pd.DataFrame:
        if not self.yolo_available or not self.opencv_available:
            raise RuntimeError(
                "YOLO/OpenCV are not installed. Install `ultralytics` and "
                "`opencv-python`, or use build_counts_from_simulation() for "
                "the included runnable prototype."
            )
        raise NotImplementedError(
            "This adapter is ready for a real YOLO implementation. For this "
            "assignment prototype, simulation-derived counts are used."
        )

    def privacy_policy(self) -> Dict[str, str]:
        return {
            "video_storage": "Do not persist raw video frames in the prototype.",
            "anonymization": "Store aggregate lane counts only; no faces or license plates.",
            "retention": "Keep CSV metrics needed for testing/reporting, not identifiable footage.",
        }


def build_counts_from_simulation(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create required vehicle-count CSV columns from simulator metrics."""

    grouped = (
        metrics.groupby(["step", "lane_id"], as_index=False)
        .agg(vehicle_count=("vehicle_count", "sum"), emergency_count=("emergency_count", "max"))
        .rename(columns={"step": "timestamp"})
    )

    rows = []
    for _, row in grouped.iterrows():
        total = int(row["vehicle_count"])
        emergency = int(row["emergency_count"])
        non_emergency = max(total - emergency, 0)
        car_count = int(round(non_emergency * 0.68))
        bus_count = int(round(non_emergency * 0.10))
        truck_count = int(round(non_emergency * 0.12))
        motorcycle_count = max(non_emergency - car_count - bus_count - truck_count, 0)
        rows.append(
            {
                "timestamp": int(row["timestamp"]),
                "lane_id": row["lane_id"],
                "car_count": car_count,
                "bus_count": bus_count,
                "truck_count": truck_count,
                "motorcycle_count": motorcycle_count,
                "emergency_count": emergency,
            }
        )
    return pd.DataFrame(rows, columns=VEHICLE_COLUMNS)
