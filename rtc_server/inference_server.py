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

"""ZMQ REQ/REP inference server for Real-Time Chunking.

Mirrors the GR00T ``RobotInferenceServer`` design: a single-threaded REP loop
dispatches named endpoints. The ``RTCInferenceServer`` registers ``get_action``
(returns a single, blended action per call) and ``reset`` (rebuilds the RTC
controller). The wire format matches the GR00T client (see ``serialization``),
so the existing counterstrike harness can drive this server unchanged.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

try:
    import zmq
except ImportError as e:  # pragma: no cover - dependency hint
    raise ImportError(
        "The RTC inference server requires `pyzmq`. Install it into the lerobot "
        "environment with `uv pip install pyzmq` (or `uv sync --extra pyzmq-dep`)."
    ) from e

from .rtc_controller import RTCController
from .rtc_policy import RTCPi05Policy
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
        logger.info("RTC inference server listening on tcp://%s:%d", self.host, self.port)
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

        logger.info("RTC inference server stopped.")

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


class RTCInferenceServer(BaseInferenceServer):
    """Serves single, RTC-blended actions from a pi0.5 policy."""

    def __init__(
        self,
        policy: RTCPi05Policy,
        fps: float = 60.0,
        queue_threshold: int = 25,
        host: str = "*",
        port: int = 5555,
        api_token: str | None = None,
    ):
        super().__init__(host=host, port=port, api_token=api_token)
        self.policy = policy
        self.fps = fps
        self.queue_threshold = queue_threshold

        self.controller = self._make_controller()
        self.register_endpoint("get_action", self._get_action)
        self.register_endpoint("reset", self._reset, requires_input=False)

    def _make_controller(self) -> RTCController:
        return RTCController(self.policy, fps=self.fps, queue_threshold=self.queue_threshold)

    def _get_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.controller.step(obs)

    def _reset(self) -> dict:
        self.controller.stop()
        self.controller = self._make_controller()
        return {"status": "reset"}

    def stop(self) -> None:
        self.controller.stop()
        super().stop()
