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

"""Entry point: serve a custom pi0.5 checkpoint with Real-Time Chunking.

Example:
```shell
uv run --active python -m rtc_server.inference_script \
    --model_path=/home/innovation-hacking/data/bozzetti/models/counterstrike/lerobot_pi05_test \
    --host="*" \
    --port=5555 \
    --device=cuda \
    --fps=60 \
    --execution_horizon=10 \
    --max_guidance_weight=10.0 \
    --prefix_attention_schedule=exp
```

The server speaks the same ZMQ + msgpack contract as the GR00T RTC server, so
the existing counterstrike harness (``inference/counterstrike/rtc_inference.py``)
can drive it after pointing the client at this server's host/port.
"""

from __future__ import annotations

import argparse
import logging

from .inference_server import RTCInferenceServer
from .rtc_policy import RTCPi05Policy, build_rtc_config

DEATHMATCH_PROMPT = (
    "You are playing counterstrike. You are a counter-terrorist and playing team deathmatch. "
    "Search and shoot the enemies."
)
ARMS_RACE_PROMPT = (
    "You are playing counterstrike. You are a counter-terrorist and playing Arms Race. "
    "Friends are marked with blue text above their heads. Search and shoot the enemies."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RTC async inference server for a LeRobot pi0.5 checkpoint.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pi0.5 checkpoint directory.")
    parser.add_argument("--host", type=str, default="*", help="ZMQ bind host ('*' for all interfaces).")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device (cuda/cpu).")
    parser.add_argument("--task", type=str, default=DEATHMATCH_PROMPT, help="Default task/language prompt.")
    parser.add_argument("--fps", type=float, default=60.0, help="Control-loop frequency (used for RTC delay).")
    parser.add_argument("--execution_horizon", type=int, default=10, help="RTC execution horizon (steps).")
    parser.add_argument("--max_guidance_weight", type=float, default=10.0, help="RTC max guidance weight.")
    parser.add_argument(
        "--prefix_attention_schedule",
        type=str,
        default="exp",
        choices=["exp", "linear", "ones", "zeros"],
        help="RTC prefix attention schedule.",
    )
    parser.add_argument(
        "--queue_threshold",
        type=int,
        default=25,
        help="Regenerate the next chunk once the action queue drops to this size.",
    )
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
        help="Enable torch.compile (forced to a non-CUDA-graph mode; off by default since "
        "RTC's autograd-in-denoise-loop is incompatible with CUDA graphs).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # lerobot configures the root logger at import time, so basicConfig above is a
    # no-op; force INFO explicitly so our readiness/latency lines are visible.
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    rtc_config = build_rtc_config(
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=args.prefix_attention_schedule,
    )

    policy = RTCPi05Policy(
        model_path=args.model_path,
        rtc_config=rtc_config,
        device=args.device,
        task=args.task,
        image_key=args.image_key,
        compile_model=args.compile,
    )

    server = RTCInferenceServer(
        policy,
        fps=args.fps,
        queue_threshold=args.queue_threshold,
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
