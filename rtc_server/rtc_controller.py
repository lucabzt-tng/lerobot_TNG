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

"""Server-side Real-Time Chunking controller.

A background thread continuously regenerates action chunks via
``policy.predict_action_chunk(..., inference_delay, prev_chunk_left_over)`` and
merges them into a thread-safe :class:`ActionQueue`. Each ``step(obs)`` call
publishes the latest observation and pops the next single, already-blended
action — mirroring the GR00T ``RTCController`` semantics but built on LeRobot's
RTC primitives (``ActionQueue`` / ``RTCProcessor`` / ``LatencyTracker``).

This is the network-server analogue of LeRobot's in-process
``RTCInferenceEngine`` (``src/lerobot/rollout/inference/rtc.py``): the
inference loop and queue management are the same; the differences are that
observations arrive as counterstrike dicts over the wire and that ``step`` is
guaranteed to return an action (it bootstraps / falls back to a synchronous
generation when the queue runs dry).
"""

from __future__ import annotations

import logging
import math
import time
from threading import Event, Lock, Thread

import torch

from lerobot.policies.rtc import ActionQueue, LatencyTracker

from .rtc_policy import RTCPi05Policy

logger = logging.getLogger(__name__)

# How long the background loop sleeps when idle (no obs) or the queue is full.
_IDLE_SLEEP_S: float = 0.005
# Backoff between transient inference errors.
_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before the loop gives up.
_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the background thread on stop().
_JOIN_TIMEOUT_S: float = 3.0


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate the RTC prefix to a fixed length for stable inference."""
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


class RTCController:
    """Produces single RTC-blended actions from a streaming observation."""

    def __init__(
        self,
        policy: RTCPi05Policy,
        fps: float = 60.0,
        queue_threshold: int = 25,
    ):
        self.policy = policy
        self.fps = fps
        self.time_per_chunk = 1.0 / fps
        self.queue_threshold = queue_threshold

        self.rtc_config = policy.policy.config.rtc_config
        self.queue = ActionQueue(self.rtc_config)
        self.latency = LatencyTracker()

        self._obs: dict | None = None
        self._obs_lock = Lock()
        self._infer_lock = Lock()  # serializes the policy forward pass
        self._last_action: torch.Tensor | None = None
        self._chunks_generated = 0

        # Compile / warm up the model before serving so the first real inference
        # is fast and its latency does not corrupt the RTC delay estimate.
        try:
            self.policy.warmup()
        except Exception:
            logger.exception("Policy warmup failed; continuing (first inference may be slow).")

        self._shutdown = Event()
        self._thread = Thread(target=self._loop, daemon=True, name="RTCInference")
        self._thread.start()
        logger.info("RTC controller started (fps=%.1f, queue_threshold=%d)", fps, queue_threshold)

    # ------------------------------------------------------------------
    # Public API (called from the server's request thread)
    # ------------------------------------------------------------------

    def step(self, obs: dict) -> dict:
        """Publish ``obs`` and return the next single blended action as a dict."""
        with self._obs_lock:
            self._obs = obs

        # On the first tick (or if the background loop has fallen behind and the
        # queue ran dry), generate synchronously so we always return an action.
        if self.queue.empty():
            self._generate(obs)

        action = self.queue.get()
        if action is None:
            # Queue still empty (e.g. generation produced nothing) -> hold last action.
            if self._last_action is None:
                raise RuntimeError("RTC controller could not produce an action for the first observation.")
            action = self._last_action

        self._last_action = action
        return self.policy.action_to_dict(action)

    def stop(self) -> None:
        self._shutdown.set()
        if self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _JOIN_TIMEOUT_S)
        logger.info("RTC controller stopped.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _generate(self, obs: dict | None = None) -> None:
        """Generate one chunk and merge it into the queue (thread-safe)."""
        with self._infer_lock:
            # Re-check under the lock: another caller may have just refilled it.
            if obs is not None and not self.queue.empty():
                return

            if obs is None:
                with self._obs_lock:
                    obs = self._obs
            if obs is None:
                return

            idx_before = self.queue.get_action_index()
            prev_actions = self.queue.get_left_over()

            latency = self.latency.max()
            delay = math.ceil(latency / self.time_per_chunk) if latency else 0

            if prev_actions is not None:
                prev_actions = _normalize_prev_actions_length(
                    prev_actions, target_steps=self.rtc_config.execution_horizon
                )

            start = time.perf_counter()
            original, processed = self.policy.predict_chunk(
                obs, inference_delay=delay, prev_chunk_left_over=prev_actions
            )
            new_latency = time.perf_counter() - start
            self.latency.add(new_latency)

            # First chunk: the robot/gamepad was not executing anything while we
            # generated it, so nothing should be skipped. For later chunks, skip
            # the actions consumed during inference, but never drop the whole
            # chunk (clamp below chunk length) so the queue can't run dry.
            chunk_len = processed.shape[0]
            if self._chunks_generated == 0:
                merge_delay = 0
            else:
                merge_delay = min(math.ceil(new_latency / self.time_per_chunk), chunk_len - 1)

            self.queue.merge(original, processed, merge_delay, idx_before)
            self._chunks_generated += 1

            logger.debug(
                "RTC inference latency=%.3fs guidance_delay=%d merge_delay=%d queue=%d",
                new_latency,
                delay,
                merge_delay,
                self.queue.qsize(),
            )

    def _loop(self) -> None:
        consecutive_errors = 0
        while not self._shutdown.is_set():
            with self._obs_lock:
                have_obs = self._obs is not None
            if not have_obs or self.queue.qsize() > self.queue_threshold:
                time.sleep(_IDLE_SLEEP_S)
                continue

            try:
                self._generate()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.error(
                    "RTC inference error (%d/%d): %s: %s",
                    consecutive_errors,
                    _MAX_CONSECUTIVE_ERRORS,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    logger.error("RTC inference giving up after repeated errors.")
                    self._shutdown.set()
                    break
                time.sleep(_ERROR_RETRY_DELAY_S)
