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

"""Entry point: serve a pi0.5 checkpoint without Real-Time Chunking.

All configuration has defaults, so the launcher can run with no arguments:

```shell
uv run --active python -m inference_server.inference_script --compile
```
"""

from __future__ import annotations

import argparse
import logging

from .policy import Pi05Policy
from .server import InferenceServer

DEFAULT_MODEL_PATH = "/home/innovation-hacking/bozzetti/models/counterstrike/lerobot_pi05_test/050000/pretrained_model"

DEATHMATCH_PROMPT = (
    "You are playing counterstrike. You are a counter-terrorist and playing team deathmatch. "
    "Search and shoot the enemies."
)
ARMS_RACE_PROMPT = (
    "You are playing counterstrike. You are a counter-terrorist and playing Arms Race. "
    "Friends are marked with blue text above their heads. Search and shoot the enemies."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference server for a LeRobot pi0.5 checkpoint (no RTC).")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="pi0.5 checkpoint directory.")
    parser.add_argument("--host", type=str, default="*", help="ZMQ bind host ('*' for all interfaces).")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (cuda/cpu).")
    parser.add_argument("--task", type=str, default=DEATHMATCH_PROMPT, help="Default task/language prompt.")
    parser.add_argument(
        "--image_key",
        type=str,
        default="observation.images._view",
        help="Dataset image feature key the policy was trained with.",
    )
    parser.add_argument("--api_token", type=str, default=None, help="Optional shared-secret API token.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (forced to a non-CUDA-graph mode, since the denoising "
        "loop's in-place updates are incompatible with CUDA graphs).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # lerobot configures the root logger at import time, so basicConfig above is a
    # no-op; force INFO explicitly so our readiness/latency lines are visible.
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    policy = Pi05Policy(
        model_path=args.model_path,
        device=args.device,
        task=args.task,
        image_key=args.image_key,
        compile_model=args.compile,
    )

    server = InferenceServer(
        policy,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
