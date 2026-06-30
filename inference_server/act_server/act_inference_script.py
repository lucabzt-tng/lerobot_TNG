"""Entry point: serve an ACT checkpoint as a synchronous chunk server.

Run from the repo root:

    uv run --active python -m inference_server.act_server.act_inference_script \\
        --model_path /home/innovation-hacking/bozzetti/models/counterstrike/act_benchmark/050000/pretrained_model
"""

from __future__ import annotations

import argparse
import logging
import time

from .act_policy import ACTPolicy, NUM_AXES
from ..server import BaseInferenceServer
from ..serialization import MsgSerializer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    "/home/innovation-hacking/bozzetti/models/counterstrike/act_benchmark"
    "/050000/pretrained_model"
)


class ACTInferenceServer(BaseInferenceServer):
    """Wraps ACTPolicy behind a ZMQ REP server.

    Each ``get_action_chunk`` call blocks while the policy runs inference and
    returns the complete predicted chunk. The client executes actions one-by-one
    and re-queries when the chunk is exhausted.
    """

    def __init__(self, policy: ACTPolicy, host: str = "*", port: int = 5556):
        super().__init__(host=host, port=port)
        self.policy = policy
        self._infer_count = 0
        self.register_endpoint("get_action_chunk", self._get_action_chunk)
        self.register_endpoint("reset", self._reset, requires_input=False)

    def _get_action_chunk(self, obs: dict) -> dict:
        start = time.perf_counter()
        chunk = self.policy.predict_chunk(obs)  # (chunk_size, action_dim)
        ms = (time.perf_counter() - start) * 1000.0
        self._infer_count += 1

        import numpy as np
        chunk_np = chunk.detach().cpu().float().numpy() if hasattr(chunk, "detach") else chunk
        logger.info("Inference #%d: %.1f ms | chunk=%s", self._infer_count, ms, tuple(chunk_np.shape))

        return {
            "action.axes": chunk_np[:, :NUM_AXES].astype("float32"),
            "action.buttons": chunk_np[:, NUM_AXES:].astype("float32"),
        }

    def _reset(self) -> dict:
        return {"status": "reset"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT inference server (synchronous chunk server).")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", type=str, default="*")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--image_key",
        type=str,
        default="observation.images._view",
        help="Dataset image feature key the policy was trained with.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()

    policy = ACTPolicy(
        model_path=args.model_path,
        device=args.device,
        image_key=args.image_key,
    )
    policy.warmup()

    server = ACTInferenceServer(policy=policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
