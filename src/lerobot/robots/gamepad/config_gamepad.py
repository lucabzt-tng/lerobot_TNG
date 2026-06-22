from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("gamepad")
@dataclass
class GamepadConfig(RobotConfig):
    # Resolution of the game capture (H, W).
    camera_height: int = 512
    camera_width: int = 512
