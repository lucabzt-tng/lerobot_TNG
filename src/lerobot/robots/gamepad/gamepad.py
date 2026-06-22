from functools import cached_property

from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation

from .config_gamepad import GamepadConfig


class Gamepad(Robot):
    """
    Gamepad declared as a LeRobot robot for training purposes.

    State / action space: 6 continuous axes + 16 buttons = 22 floats.
    Observation also includes one game-capture camera keyed "_view".
    """

    config_class = GamepadConfig
    name = "gamepad"

    def __init__(self, config: GamepadConfig):
        super().__init__(config)
        self.config = config

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for i in range(6):
            features[f"axes_{i}"] = float
        for i in range(16):
            features[f"button_{i}"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self.action_features,
            "_view": (self.config.camera_height, self.config.camera_width, 3),
        }

    # ------------------------------------------------------------------
    # Stubs — not needed for training from an existing dataset
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return False

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        raise NotImplementedError

    def send_action(self, action: RobotAction) -> RobotAction:
        raise NotImplementedError
