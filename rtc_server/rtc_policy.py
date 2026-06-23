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

"""Thin RTC-aware wrapper around a LeRobot pi0.5 policy.

Handles three things the RTC controller should not care about:
- loading the checkpoint and enabling :class:`RTCConfig` on it,
- converting the counterstrike observation dict (``video._view`` +
  ``state.axes`` / ``state.buttons`` + task prompt) into a model-ready batch,
- mapping the policy's flat action vector back to ``action.axes`` /
  ``action.buttons`` for the game client.

The wire contract matches ``inference/counterstrike/inference_client.py`` so the
existing harness drives this server unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_STATE

logger = logging.getLogger(__name__)

# The counterstrike state/action vector is [6 stick/trigger axes, then buttons].
NUM_AXES = 6


class RTCPi05Policy:
    """Loads a pi0.5 checkpoint with RTC enabled and exposes a chunk predictor."""

    def __init__(
        self,
        model_path: str,
        rtc_config: RTCConfig,
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

        # RTC's guidance runs torch.autograd.grad *inside* the denoising loop, which
        # cannot be captured by CUDA graphs. Checkpoints trained with
        # compile_mode='reduce-overhead' (CUDA graphs) crash under RTC with
        # cudaErrorStreamCaptureInvalidated. Disable compilation for RTC inference by
        # default; if explicitly enabled, force a compile mode without CUDA graphs.
        if compile_model:
            config.compile_model = True
            if getattr(config, "compile_mode", None) in ("reduce-overhead", "max-autotune"):
                logger.warning(
                    "Overriding compile_mode=%s -> 'default' (CUDA graphs are incompatible with RTC).",
                    config.compile_mode,
                )
                config.compile_mode = "default"
        else:
            config.compile_model = False

        policy_class = get_policy_class(config.type)
        self.policy = policy_class.from_pretrained(model_path, config=config)
        self.policy.to(device)
        self.policy.eval()

        # Enable Real-Time Chunking on the loaded policy.
        self.policy.config.rtc_config = rtc_config
        self.policy.init_rtc_processor()
        logger.info(
            "Loaded %s from %s with RTC enabled "
            "(execution_horizon=%d, max_guidance_weight=%.2f, schedule=%s)",
            config.type,
            model_path,
            rtc_config.execution_horizon,
            rtc_config.max_guidance_weight,
            rtc_config.prefix_attention_schedule.value,
        )

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
    def predict_chunk(
        self,
        obs: dict,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one RTC-guided chunk prediction.

        Returns ``(original, processed)`` where ``original`` is the normalized
        chunk (fed back as the next RTC prefix) and ``processed`` is the
        unnormalized chunk ready for the robot/gamepad. Both have shape
        ``(chunk_size, action_dim)``.
        """
        batch = self.build_batch(obs)
        actions = self.policy.predict_action_chunk(
            batch,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
        )
        original = actions.squeeze(0).clone()
        processed = self.postprocessor(actions).squeeze(0)
        return original, processed

    def warmup(self, num_inferences: int = 2) -> None:
        """Run a few throwaway inferences on synthetic input to trigger
        ``torch.compile`` / CUDA-graph capture before the server starts serving.

        Without this, the first real ``get_action`` blocks for the full
        compilation time (tens of seconds), the client times out, and the huge
        measured latency corrupts the RTC delay estimate. We exercise both the
        first-chunk path (``prev_chunk_left_over=None``) and the steady-state
        path (a non-empty prefix) so both compiled graphs are cached.
        """
        h, w = self.image_hw
        obs = {
            "state.axes": np.zeros(NUM_AXES, dtype=np.float32),
            "state.buttons": np.zeros(self.state_dim - NUM_AXES, dtype=np.float32),
            "video._view": np.zeros((1, h, w, 3), dtype=np.uint8),
            "annotation.human.task_description": [self.task],
        }
        execution_horizon = self.policy.config.rtc_config.execution_horizon
        prefix = torch.zeros(execution_horizon, self.action_dim, device=self.device)

        logger.warning("Warming up policy (compiling) with %d inference(s)...", num_inferences)
        for i in range(max(1, num_inferences)):
            prev = None if i == 0 else prefix
            self.predict_chunk(obs, inference_delay=0, prev_chunk_left_over=prev)
        logger.warning("Policy warmup complete.")


def build_rtc_config(
    execution_horizon: int = 10,
    max_guidance_weight: float = 10.0,
    prefix_attention_schedule: str | RTCAttentionSchedule = RTCAttentionSchedule.EXP,
) -> RTCConfig:
    """Helper to build an enabled :class:`RTCConfig` from CLI-friendly values."""
    if isinstance(prefix_attention_schedule, str):
        prefix_attention_schedule = RTCAttentionSchedule(prefix_attention_schedule.upper())
    return RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=max_guidance_weight,
        prefix_attention_schedule=prefix_attention_schedule,
    )
