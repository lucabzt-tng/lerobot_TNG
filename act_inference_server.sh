#!/usr/bin/env bash
# Launch the ACT synchronous chunk inference server.
#
# Usage:
#   ./act_inference_server.sh [--model_path PATH] [--port PORT] [--device DEVICE]
#
# Defaults to the last checkpoint from train_act_counterstrike.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="/home/innovation-hacking/bozzetti/models/counterstrike/act_benchmark/checkpoints/050000/pretrained_model"
PORT=5556
DEVICE=cuda

# Pass through any extra args directly to the Python script
uv run --active python -m inference_server.act_server.act_inference_script \
    --model_path "$MODEL_PATH" \
    --port "$PORT" \
    --device "$DEVICE" \
    "$@"
