"""Quick test: DQN on SUMO GUI (10 steps)."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from control.dqn_controller import DQNController, controller_policy
from simulation.sumo_env import SumoTrafficEnvironment
from simulation.synthetic_sumo import TrafficEnvironment


def main() -> None:
    sample_env = TrafficEnvironment(scenario="normal", steps=30, seed=7)
    controller = DQNController(sample_env.state_size, sample_env.action_size, seed=17)

    for ep in range(5):
        env = TrafficEnvironment(
            scenario=["normal", "peak", "emergency", "accident"][ep % 4],
            steps=20, seed=7 + ep,
        )
        state = env.reset()
        done = False
        while not done:
            a, r = controller.select_action(state, explore=True)
            res = env.step(a, reason=r)
            controller.remember(state, a, res.reward, res.state, res.done)
            controller.train_batch(32)
            state = res.state
            done = res.done
        controller.decay_epsilon()
        print(f"Episode {ep + 1} done")

    controller.epsilon = 0.0
    policy = controller_policy(controller)

    print("Launching SUMO...")
    env = SumoTrafficEnvironment(scenario="normal", steps=10, gui=False, seed=42)
    state = env.reset()
    for i in range(10):
        import time
        t0 = time.perf_counter()
        a, r = policy(state, env)
        latency = time.perf_counter() - t0
        res = env.step(a, reason=r, decision_latency=latency)
        state = res.state
        gl = res.info["green_lane"]
        rw = res.reward
        print(f"  Step {i}: phase={gl}, reward={rw:.1f}")

    env._stop_sumo()
    print("SUMO test complete.")


if __name__ == "__main__":
    main()
