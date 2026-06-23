# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Non-RTC pi0.5 policy wrapper for the standard inference server.

Loads a pi0.5 checkpoint and exposes a plain ``predict_chunk(obs)`` that runs
``predict_action_chunk`` with no Real-Time Chunking guidance. Observation /
action conversion uses the same counterstrike wire format as ``rtc_server``
(``video._view`` + ``state.axes`` / ``state.buttons`` -> a single 22-dim
``observation.state`` / ``action`` vector).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

# The counterstrike state/action vector is [6 stick/trigger axes, then buttons].
NUM_AXES = 6


class Pi05Policy:
    """Loads a pi0.5 checkpoint (RTC disabled) and predicts action chunks."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        task: str = "",
        image_key: str = "observation.images._view",
        image_hw: tuple[int, int] = (512, 512),
        compile_model: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.task = task
        self.image_key = image_key
        self.image_hw = image_hw

        config = PreTrainedConfig.from_pretrained(model_path)
        config.pretrained_path = model_path

        # The pi0.5 denoising loop uses in-place updates (x_t += dt * v_t) that are
        # incompatible with CUDA-graph capture, so a checkpoint trained with
        # compile_mode='reduce-overhead' crashes when compiled. Disable compilation
        # by default; if enabled, force a compile mode without CUDA graphs.
        if compile_model:
            config.compile_model = True
            if getattr(config, "compile_mode", None) in ("reduce-overhead", "max-autotune"):
                logger.warning(
                    "Overriding compile_mode=%s -> 'default' (CUDA graphs break the denoising loop).",
                    config.compile_mode,
                )
                config.compile_mode = "default"
        else:
            config.compile_model = False

        # Ensure no RTC guidance is active.
        config.rtc_config = None

        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(model_path, config=config)
        self.policy.to(device)
        self.policy.eval()
        logger.info("Loaded %s from %s (RTC disabled, compile=%s)", config.type, model_path, compile_model)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )

        self.action_dim = self.policy.config.output_features[ACTION].shape[0]
        self.state_dim = self.policy.config.input_features[OBS_STATE].shape[0]

    # ------------------------------------------------------------------
    # Observation / action conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _as_image_hwc(raw: np.ndarray) -> np.ndarray:
        """Squeeze leading batch/time dims down to a single ``(H, W, 3)`` frame."""
        arr = np.squeeze(np.asarray(raw))
        if arr.ndim == 4:  # (T, H, W, C) -> most recent frame
            arr = arr[-1]
        if arr.ndim != 3:
            raise ValueError(f"Expected an (H, W, C) image, got shape {np.asarray(raw).shape}")
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

    def _resolve_task(self, obs: dict) -> str:
        prompt = obs.get("annotation.human.task_description")
        if isinstance(prompt, (list, tuple, np.ndarray)) and len(prompt) > 0:
            return str(prompt[0])
        if isinstance(prompt, str) and prompt:
            return prompt
        return self.task

    def build_batch(self, obs: dict) -> dict:
        """Convert a counterstrike observation dict into a preprocessed batch."""
        image = self._as_image_hwc(obs["video._view"])
        image_t = (
            torch.from_numpy(np.ascontiguousarray(image))
            .to(torch.float32)
            .div_(255.0)
            .permute(2, 0, 1)  # HWC -> CHW
            .unsqueeze(0)  # add batch dim
        )
        state_t = torch.from_numpy(self._build_state(obs)).unsqueeze(0)

        batch = {
            self.image_key: image_t,
            OBS_STATE: state_t,
            "task": self._resolve_task(obs),
        }
        return self.preprocessor(batch)

    def action_to_dict(self, action: torch.Tensor) -> dict[str, np.ndarray]:
        """Split a flat action vector into ``action.axes`` / ``action.buttons``."""
        a = action.detach().to("cpu").to(torch.float32).numpy()
        return {
            "action.axes": a[:NUM_AXES],
            "action.buttons": a[NUM_AXES:],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_chunk(self, obs: dict) -> torch.Tensor:
        """Predict and post-process an action chunk. Returns ``(chunk_size, action_dim)``."""
        batch = self.build_batch(obs)
        actions = self.policy.predict_action_chunk(batch)
        return self.postprocessor(actions).squeeze(0)

    def warmup(self, num_inferences: int = 2) -> None:
        """Run a few throwaway inferences on synthetic input to trigger
        ``torch.compile`` / kernel autotuning before serving, so the first real
        ``get_action`` is fast."""
        h, w = self.image_hw
        obs = {
            "state.axes": np.zeros(NUM_AXES, dtype=np.float32),
            "state.buttons": np.zeros(self.state_dim - NUM_AXES, dtype=np.float32),
            "video._view": np.zeros((1, h, w, 3), dtype=np.uint8),
            "annotation.human.task_description": [self.task],
        }
        logger.warning("Warming up policy with %d inference(s)...", num_inferences)
        for _ in range(max(1, num_inferences)):
            self.predict_chunk(obs)
        logger.warning("Policy warmup complete.")
