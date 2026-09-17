"""Run DQN controller on SUMO GUI — observable in real time.

Controls while SUMO GUI is open:
  Space / Play button  — pause / resume
  F3                  — faster
  F4                  — slower
  Close the window    — stop
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control.dqn_controller import DQNController, controller_policy
from simulation.sumo_env import SumoTrafficEnvironment
from simulation.synthetic_sumo import TrafficEnvironment

from traci.exceptions import FatalTraCIError


def main() -> None:
    print("Step 1: Training DQN controller on synthetic environment...")
    sample_env = TrafficEnvironment(scenario="normal", steps=100, seed=7)
    controller = DQNController(sample_env.state_size, sample_env.action_size, seed=17)

    scenarios = ["normal", "peak", "emergency", "accident"]
    for episode in range(20):
        env = TrafficEnvironment(
            scenario=scenarios[episode % 4], steps=60, seed=7 + episode
        )
        state = env.reset()
        done = False
        while not done:
            action, reason = controller.select_action(state, explore=True)
            result = env.step(action, reason=reason)
            controller.remember(state, action, result.reward, result.state, result.done)
            controller.train_batch(batch_size=32)
            state = result.state
            done = result.done
        controller.decay_epsilon()
        print(f"  Episode {episode + 1}: {scenarios[episode % 4]}")

    controller.epsilon = 0.0
    policy = controller_policy(controller)

    print("\nStep 2: Launching SUMO GUI — emergency scenario")
    print("  Normal vehicles = WHITE      Emergency vehicle = RED")
    print("  Speed controls: Space=pause, F3=faster, F4=slower")
    print("  Watch for the red emergency vehicle at step 24 on east lane\n")

    env = SumoTrafficEnvironment(scenario="emergency", steps=60, gui=True, seed=42)
    state = env.reset()
    done = False
    try:
        while not done:
            action, reason = policy(state, env)
            result = env.step(action, reason=reason)
            state = result.state
            done = result.done
            step_idx = env._step_index - 1
            phase = result.info["green_lane"]
            rw = result.reward
            print(f"  Step {step_idx:2d}/{env.steps}  phase={phase:6s}  reward={rw:6.1f}")
            time.sleep(0.8)
    except (FatalTraCIError, ConnectionError, OSError):
        print("\nSUMO GUI closed by user.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            env._stop_sumo()
        except Exception:
            pass

    metrics = env.metrics_frame()
    em_rows = metrics[metrics["emergency_count"] > 0]
    print(f"\nEmergency vehicle detected on {len(em_rows)} steps")
    ct = env.emergency_clearance_time()
    if ct:
        print(f"Clearance time: {ct} seconds")
    else:
        print("Emergency was NOT cleared by the AI controller.")
    print("Done.")


if __name__ == "__main__":
    main()
