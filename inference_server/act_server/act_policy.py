"""ACT policy wrapper for the inference server.

Loads an ACT checkpoint and exposes ``predict_chunk(obs)`` which runs
``predict_action_chunk`` and returns a ``(chunk_size, action_dim)`` tensor.
Observation / action conversion uses the same counterstrike wire format as the
pi0.5 server (``video._view`` + ``state.axes`` / ``state.buttons`` -> a single
22-dim ``observation.state`` / ``action`` vector).

ACT has no language conditioning and no RTC, so neither is wired up here.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

NUM_AXES = 6


class ACTPolicy:
    """Loads an ACT checkpoint and predicts full action chunks."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        image_key: str = "observation.images._view",
        image_hw: tuple[int, int] = (512, 512),
    ):
        self.model_path = model_path
        self.device = device
        self.image_key = image_key
        self.image_hw = image_hw

        config = PreTrainedConfig.from_pretrained(model_path)
        config.pretrained_path = model_path

        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(model_path, config=config)
        self.policy.to(device)
        self.policy.eval()
        logger.info("Loaded %s from %s on %s", config.type, model_path, device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )

        self.action_dim = self.policy.config.output_features[ACTION].shape[0]
        self.state_dim = self.policy.config.input_features[OBS_STATE].shape[0]

    @staticmethod
    def _as_image_hwc(raw: np.ndarray) -> np.ndarray:
        arr = np.squeeze(np.asarray(raw))
        if arr.ndim == 4:
            arr = arr[-1]
        if arr.ndim != 3:
            raise ValueError(f"Expected (H, W, C) image, got shape {np.asarray(raw).shape}")
        return arr

    def _build_state(self, obs: dict) -> np.ndarray:
        axes = np.asarray(obs["state.axes"], dtype=np.float32).reshape(-1)[:NUM_AXES]
        buttons = np.asarray(obs["state.buttons"], dtype=np.float32).reshape(-1)
        state = np.concatenate([axes, buttons]).astype(np.float32)
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"State dim mismatch: built {state.shape[0]} "
                f"(axes={axes.shape[0]} + buttons={buttons.shape[0]}) but policy expects {self.state_dim}"
            )
        return state

    def build_batch(self, obs: dict) -> dict:
        image = self._as_image_hwc(obs["video._view"])
        image_t = (
            torch.from_numpy(np.ascontiguousarray(image))
            .to(torch.float32)
            .div_(255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        state_t = torch.from_numpy(self._build_state(obs)).unsqueeze(0)

        batch = {
            self.image_key: image_t,
            OBS_STATE: state_t,
        }
        return self.preprocessor(batch)

    def action_to_dict(self, action: torch.Tensor) -> dict[str, np.ndarray]:
        a = action.detach().to("cpu").to(torch.float32).numpy()
        return {
            "action.axes": a[:NUM_AXES],
            "action.buttons": a[NUM_AXES:],
        }

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> torch.Tensor:
        """Predict and post-process an action chunk. Returns ``(chunk_size, action_dim)``."""
        batch = self.build_batch(obs)
        actions = self.policy.predict_action_chunk(batch)
        return self.postprocessor(actions).squeeze(0)

    def warmup(self, num_inferences: int = 2) -> None:
        h, w = self.image_hw
        obs = {
            "state.axes": np.zeros(NUM_AXES, dtype=np.float32),
            "state.buttons": np.zeros(self.state_dim - NUM_AXES, dtype=np.float32),
            "video._view": np.zeros((1, h, w, 3), dtype=np.uint8),
        }
        logger.warning("Warming up ACT policy with %d inference(s)...", num_inferences)
        for _ in range(max(1, num_inferences)):
            self.predict_chunk(obs)
        logger.warning("ACT policy warmup complete.")
