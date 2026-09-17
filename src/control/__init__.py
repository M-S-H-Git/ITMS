
try:
    from .dqn_controller import DQNController, controller_policy, train_dqn_controller
except ImportError:
    from dqn_controller import DQNController, controller_policy, train_dqn_controller

__all__ = ["DQNController", "controller_policy", "train_dqn_controller"]
