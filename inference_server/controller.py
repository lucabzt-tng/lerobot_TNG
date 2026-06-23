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

"""Synchronous chunk controller.

Each ``get_chunk`` call runs inference on the given observation and returns the
full predicted action chunk. Chunk execution and re-querying are handled
entirely by the client; the server has no per-step state.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from .policy import NUM_AXES, Pi05Policy

logger = logging.getLogger(__name__)


class ChunkController:
    """Synchronous inference: each call blocks until the policy returns a
    complete action chunk, then returns all actions in one response."""

    def __init__(self, policy: Pi05Policy, warmup: bool = True):
        self.policy = policy
        self._infer_count = 0

        if warmup:
            try:
                self.policy.warmup()
            except Exception:
                logger.exception("Policy warmup failed; continuing (first inference may be slow).")

        logger.info("Chunk controller ready (synchronous, returns full action chunk per call)")

    def get_chunk(self, obs: dict) -> dict:
        """Run inference on *obs* and return the full action chunk.

        Returns:
            dict with keys:
                ``"action.axes"``    — ``np.ndarray`` of shape ``(chunk_size, NUM_AXES)``
                ``"action.buttons"`` — ``np.ndarray`` of shape ``(chunk_size, num_buttons)``
        """
        start = time.perf_counter()
        chunk = self.policy.predict_chunk(obs)  # (chunk_size, action_dim) tensor
        inference_ms = (time.perf_counter() - start) * 1000.0

        self._infer_count += 1
        chunk_np = chunk.detach().cpu().float().numpy() if hasattr(chunk, "detach") else np.asarray(chunk, dtype=np.float32)

        logger.info(
            "Inference #%d: %.1f ms | chunk shape=%s",
            self._infer_count,
            inference_ms,
            tuple(chunk_np.shape),
        )

        return {
            "action.axes": chunk_np[:, :NUM_AXES].astype(np.float32),
            "action.buttons": chunk_np[:, NUM_AXES:].astype(np.float32),
        }

    def reset(self) -> None:
        pass

    def stop(self) -> None:
        pass
