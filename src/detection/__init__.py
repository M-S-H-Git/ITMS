"""Vehicle detection and counting modules."""

try:
    from .vehicle_counter import VehicleCounter, build_counts_from_simulation
except ImportError:
    from vehicle_counter import VehicleCounter, build_counts_from_simulation

__all__ = ["VehicleCounter", "build_counts_from_simulation"]
