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

"""ZMQ REQ/REP inference server (no Real-Time Chunking).

Same wire contract as the GR00T / RTC servers: a single-threaded REP loop
dispatches named, msgpack-serialized endpoints. ``InferenceServer`` registers
``get_action_chunk`` (full action chunk per call, from a synchronous chunk
controller) and ``reset``. Chunk execution and re-querying are managed by the
client.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

try:
    import zmq
except ImportError as e:  # pragma: no cover - dependency hint
    raise ImportError(
        "The inference server requires `pyzmq`. Install it into the lerobot "
        "environment with `uv pip install pyzmq`."
    ) from e

from .controller import ChunkController
from .policy import Pi05Policy
from .serialization import MsgSerializer

logger = logging.getLogger(__name__)


class BaseInferenceServer:
    """A minimal ZMQ REP server with named, msgpack-serialized endpoints."""

    def __init__(self, host: str = "*", port: int = 5555, api_token: str | None = None):
        self.host = host
        self.port = port
        self.api_token = api_token
        self._running = False

        self._endpoints: dict[str, tuple[Callable, bool]] = {}
        self.register_endpoint("ping", self._ping, requires_input=False)
        self.register_endpoint("kill", self._kill, requires_input=False)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True) -> None:
        """Register ``handler`` under ``name``. ``handler`` takes the request data
        (a dict) when ``requires_input`` else no args, and returns a dict."""
        self._endpoints[name] = (handler, requires_input)

    def _ping(self) -> dict:
        return {"status": "ok"}

    def _kill(self) -> dict:
        self._running = False
        return {"status": "stopping"}

    def run(self) -> None:
        self._running = True
        logger.info("Inference server listening on tcp://%s:%d", self.host, self.port)
        while self._running:
            try:
                message = self.socket.recv()
            except zmq.ContextTerminated:
                break

            try:
                request = MsgSerializer.from_bytes(message)
                if self.api_token and request.get("api_token") != self.api_token:
                    response = {"error": "Invalid or missing API token."}
                else:
                    response = self._dispatch(request)
            except Exception as e:
                logger.error("Error handling request: %s", e)
                logger.debug(traceback.format_exc())
                response = {"error": f"{type(e).__name__}: {e}"}

            self.socket.send(MsgSerializer.to_bytes(response))

        logger.info("Inference server stopped.")

    def _dispatch(self, request: dict) -> dict:
        endpoint = request.get("endpoint")
        if endpoint not in self._endpoints:
            return {"error": f"Unknown endpoint: {endpoint}"}

        handler, requires_input = self._endpoints[endpoint]
        result = handler(request.get("data")) if requires_input else handler()
        if not isinstance(result, dict):
            return {"error": f"Endpoint '{endpoint}' must return a dict, got {type(result).__name__}"}
        return result

    def stop(self) -> None:
        self._running = False
        self.socket.close(linger=0)
        self.context.term()


class InferenceServer(BaseInferenceServer):
    """Serves full action chunks from a pi0.5 policy.

    Each ``get_action_chunk`` call blocks while the policy runs inference and
    returns the complete predicted chunk. The client is responsible for
    executing the actions one-by-one and re-querying when the chunk is
    exhausted.
    """

    def __init__(
        self,
        policy: Pi05Policy,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
    ):
        super().__init__(host=host, port=port, api_token=api_token)
        self.controller = ChunkController(policy)
        self.register_endpoint("get_action_chunk", self._get_action_chunk)
        self.register_endpoint("reset", self._reset, requires_input=False)

    def _get_action_chunk(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.controller.get_chunk(obs)

    def _reset(self) -> dict:
        self.controller.reset()
        return {"status": "reset"}

    def stop(self) -> None:
        self.controller.stop()
        super().stop()
