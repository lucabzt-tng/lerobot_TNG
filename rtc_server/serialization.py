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

"""msgpack (de)serialization for the RTC inference server.

The wire format is intentionally identical to the GR00T reference
``MsgSerializer`` (``inference/counterstrike/inference_client.py``) so the same
game-side client harness can talk to either server: numpy arrays are encoded
with ``np.save`` into a msgpack extension object keyed ``__ndarray_class__``.
"""

from __future__ import annotations

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
