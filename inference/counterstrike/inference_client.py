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

"""ZMQ client for the RTC inference server.

Drop-in compatible with the GR00T ``Gr00tInferenceClient`` used by
``inference/counterstrike/rtc_inference.py``: ``get_action(obs)`` sends a
counterstrike observation dict and returns ``{"action.axes", "action.buttons"}``.
"""

from __future__ import annotations

from typing import Any

import zmq

import io
from typing import Any

import numpy as np

try:
    import msgpack
except ImportError as e:  # pragma: no cover - dependency hint
    raise ImportError(
        "The RTC inference server requires `msgpack`. Install it into the lerobot "
        "environment with `uv pip install msgpack`."
    ) from e


class MsgSerializer:
    """Serialize/deserialize dicts (with numpy arrays) over msgpack."""

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode, raw=False)

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    @staticmethod
    def _decode(obj: dict) -> Any:
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj


class RTCInferenceClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> dict:
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(MsgSerializer.to_bytes(request))
        response = MsgSerializer.from_bytes(self.socket.recv())
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_action_chunk(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Request the full predicted action chunk from the server.

        Returns a dict with:
            ``"action.axes"``    — ``np.ndarray`` of shape ``(chunk_size, 6)``
            ``"action.buttons"`` — ``np.ndarray`` of shape ``(chunk_size, num_buttons)``
        """
        return self.call_endpoint("get_action_chunk", observations)

    def reset(self) -> dict:
        return self.call_endpoint("reset", requires_input=False)

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def __del__(self):
        try:
            self.socket.close(linger=0)
            self.context.term()
        except Exception:
            pass
