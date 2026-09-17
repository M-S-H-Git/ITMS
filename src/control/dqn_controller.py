from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Sequence, Tuple
import numpy as np

from simulation.synthetic_sumo import LANES, SCENARIOS, TrafficEnvironment


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


@dataclass
class DQNController:
    state_size: int
    action_size: int
    hidden_size: int = 24
    learning_rate: float = 0.015
    gamma: float = 0.92
    epsilon: float = 0.35
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.96
    seed: int = 11
    memory_size: int = 2500
    w1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)
    replay_buffer: Deque[Transition] = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.w1 = self.rng.normal(0, 0.18, size=(self.state_size, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = self.rng.normal(0, 0.18, size=(self.hidden_size, self.action_size))
        self.b2 = np.zeros(self.action_size)
        self.replay_buffer = deque(maxlen=self.memory_size)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        _, hidden, q_values = self._forward(state)
        return q_values

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, str]:
        emergency_slice_start = len(LANES) * 2
        emergency_flags = state[emergency_slice_start : emergency_slice_start + len(LANES)]
        if emergency_flags.max(initial=0.0) > 0:
            return int(np.argmax(emergency_flags)), "emergency_priority"
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size)), "dqn_exploration"
        return int(np.argmax(self.q_values(state))), "dqn_policy"

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay_buffer.append((state.copy(), int(action), float(reward), next_state.copy(), bool(done)))

    def train_batch(self, batch_size: int = 32) -> float:
        if len(self.replay_buffer) < batch_size:
            return 0.0
        indices = self.rng.choice(len(self.replay_buffer), size=batch_size, replace=False)
        losses: List[float] = []
        buffer_list = list(self.replay_buffer)
        for index in indices:
            state, action, reward, next_state, done = buffer_list[int(index)]
            target = self.q_values(state).copy()
            next_value = 0.0 if done else float(np.max(self.q_values(next_state)))
            target[action] = reward + self.gamma * next_value
            losses.append(self._train_single(state, target))
        return float(np.mean(losses))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = state @ self.w1 + self.b1
        hidden = np.maximum(z1, 0.0)
        q_values = hidden @ self.w2 + self.b2
        return z1, hidden, q_values

    def _train_single(self, state: np.ndarray, target: np.ndarray) -> float:
        z1, hidden, q_values = self._forward(state)
        error = q_values - target
        loss = float(np.mean(error**2))

        grad_q = 2.0 * error / self.action_size
        grad_w2 = np.outer(hidden, grad_q)
        grad_b2 = grad_q
        grad_hidden = self.w2 @ grad_q
        grad_z1 = grad_hidden * (z1 > 0)
        grad_w1 = np.outer(state, grad_z1)
        grad_b1 = grad_z1

        for grad in (grad_w1, grad_b1, grad_w2, grad_b2):
            np.clip(grad, -4.0, 4.0, out=grad)

        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        return loss


def train_dqn_controller(
    scenarios: Sequence[str] = ("normal", "peak", "emergency", "accident"),
    episodes: int = 44,
    steps: int = 100,
    seed: int = 11,
) -> Tuple[DQNController, List[dict]]:
    sample_env = TrafficEnvironment(scenario=scenarios[0], steps=steps, seed=seed)
    controller = DQNController(sample_env.state_size, sample_env.action_size, seed=seed)
    history: List[dict] = []

    for episode in range(episodes):
        scenario = scenarios[episode % len(scenarios)]
        env = TrafficEnvironment(scenario=scenario, steps=steps, seed=seed + episode)
        state = env.reset()
        total_reward = 0.0
        done = False
        losses: List[float] = []
        while not done:
            action, reason = controller.select_action(state, explore=True)
            result = env.step(action, reason=reason)
            controller.remember(state, action, result.reward, result.state, result.done)
            loss = controller.train_batch(batch_size=32)
            if loss:
                losses.append(loss)
            total_reward += result.reward
            state = result.state
            done = result.done
        controller.decay_epsilon()
        history.append(
            {
                "episode": episode + 1,
                "scenario": scenario,
                "reward": round(total_reward, 3),
                "epsilon": round(controller.epsilon, 4),
                "loss": round(float(np.mean(losses)), 5) if losses else 0.0,
            }
        )
    controller.epsilon = 0.0
    return controller, history


def controller_policy(controller: DQNController):
    def policy(state: np.ndarray, env: TrafficEnvironment) -> Tuple[int, str]:
        action, reason = controller.select_action(state, explore=False)
        if reason == "emergency_priority":
            return action, reason

        queues = state[: len(LANES)] * 30.0
        delay_pressure = state[len(LANES) : len(LANES) * 2] * 90.0
        accident_slice_start = len(LANES) * 3
        accident_flags = state[accident_slice_start : accident_slice_start + len(LANES)]

        q_values = controller.q_values(state)
        q_scale = np.std(q_values) + 1e-6
        q_tiebreaker = (q_values - np.mean(q_values)) / q_scale
        pressure_score = queues + 0.15 * delay_pressure + 0.25 * q_tiebreaker
        pressure_score[accident_flags > 0] -= 100.0

        pressure_action = int(np.argmax(pressure_score))
        if pressure_action != action:
            action = pressure_action
            reason = "dqn_queue_pressure"
        if accident_flags.max(initial=0.0) > 0:
            reason = "dqn_policy_accident_aware"
        return action, reason

    return policy
