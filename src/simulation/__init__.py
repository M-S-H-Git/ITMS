"""Simulation utilities for the ITMS prototype."""

try:
    from .synthetic_sumo import LANES, SCENARIOS, TrafficEnvironment, run_policy, run_fixed_timer
except ImportError:
    from synthetic_sumo import LANES, SCENARIOS, TrafficEnvironment, run_policy, run_fixed_timer

__all__ = ["LANES", "SCENARIOS", "TrafficEnvironment", "run_policy", "run_fixed_timer"]


if __name__ == "__main__":
    print("Simulation package loaded successfully.")
    print(f"Available scenarios: {', '.join(SCENARIOS)}")
    print("Run the full prototype with: python scripts/run_pipeline.py")
