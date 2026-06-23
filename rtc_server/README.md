# RTC Async Inference Server (LeRobot pi0.5)

Real-time, low-latency inference for a custom-trained **pi0.5** LeRobot
checkpoint using **Real-Time Chunking (RTC)**. This is the LeRobot counterpart
of the GR00T RTC server in `models/WR-Isaac-Gr00t/rtc` — same network contract
(ZMQ REQ/REP + msgpack), same "one blended action per `get_action` call"
semantics, so the existing counterstrike game harness can drive it.

## How it works

```
game client  ──get_action(obs)──▶  RTCInferenceServer (ZMQ REP)
   (60 Hz)   ◀──single action────       │
                                         ▼
                                   RTCController
                                   ├─ ActionQueue (thread-safe)
                                   └─ background thread:
                                        policy.predict_action_chunk(
                                            obs,
                                            inference_delay=…,        # from measured latency
                                            prev_chunk_left_over=…,   # unexecuted tail of previous chunk
                                        )
```

- A **background thread** regenerates the next action chunk before the queue
  drains (once `qsize <= queue_threshold`), guiding the new chunk to blend
  smoothly with the unexecuted tail of the previous one (RTC inpainting).
- `get_action(obs)` publishes the latest observation and pops the next single,
  already-blended action. The first call (and any time the queue runs dry)
  generates synchronously so a valid action is always returned.
- RTC guidance itself runs inside the policy's flow-matching denoiser
  (`lerobot.policies.rtc.RTCProcessor`), enabled via `RTCConfig`. The controller
  reuses LeRobot's `ActionQueue` and `LatencyTracker`.

This mirrors LeRobot's in-process `RTCInferenceEngine`
(`src/lerobot/rollout/inference/rtc.py`); the only differences are the ZMQ
transport and the counterstrike observation/action wire format.

## Wire format

Input observation dict (per `get_action`):

| key                                  | shape / type          | notes                              |
| ------------------------------------ | --------------------- | ---------------------------------- |
| `video._view`                        | `(H, W, 3)` uint8 RGB | leading batch/time dims are squeezed; resized to the training resolution by the policy preprocessor |
| `state.axes`                         | `(6,)` float32        | stick X/Y, triggers                |
| `state.buttons`                      | `(16,)` float32       | button states                      |
| `annotation.human.task_description`  | `list[str]`           | optional; falls back to `--task`   |

Output action dict:

| key              | shape / type    |
| ---------------- | --------------- |
| `action.axes`    | `(6,)` float32  |
| `action.buttons` | `(16,)` float32 |

Internally the policy uses a single 22-dim `observation.state` / `action`
vector = `concat(axes[0:6], buttons)`.

## Dependencies

The policy runs in the lerobot environment. The server additionally needs
`pyzmq` and `msgpack`, which are **not** installed by default:

```bash
uv pip install pyzmq msgpack
```

## Run the server

```bash
uv run --active python -m rtc_server.inference_script \
    --model_path=/home/innovation-hacking/data/bozzetti/models/counterstrike/lerobot_pi05_test \
    --host="*" --port=5555 --device=cuda \
    --fps=60 \
    --execution_horizon=10 \
    --max_guidance_weight=10.0 \
    --prefix_attention_schedule=exp
```

(Run from the `models/lerobot_TNG` directory so `rtc_server` is importable.)

### Key flags

| flag                          | default | meaning                                                        |
| ----------------------------- | ------- | -------------------------------------------------------------- |
| `--execution_horizon`         | 10      | timesteps to keep consistent with the previous chunk           |
| `--max_guidance_weight`       | 10.0    | strength of RTC consistency guidance                           |
| `--prefix_attention_schedule` | exp     | `exp` / `linear` / `ones` / `zeros`                            |
| `--fps`                       | 60      | control rate; converts measured latency → integer chunk delay  |
| `--queue_threshold`           | 25      | regenerate the next chunk once the queue drops to this size    |

## Client

Use the bundled `RTCInferenceClient` (drop-in for the GR00T
`Gr00tInferenceClient`):

```python
from rtc_server.client import RTCInferenceClient

client = RTCInferenceClient(host="127.0.0.1", port=5555)
action = client.get_action(observation_dict)   # {"action.axes": ..., "action.buttons": ...}
client.reset()                                  # clear RTC state between episodes
```

The existing `inference/counterstrike/rtc_inference.py` harness works against
this server by swapping `Gr00tInferenceClient` for `RTCInferenceClient`.

## Notes

- The checkpoint is trained without audio (`v9_no_audio`); audio fields in the
  observation dict are ignored.
- `reset` tears down and rebuilds the RTC controller — call it between rounds.
- **torch.compile / RTC:** compilation is **disabled by default**. RTC guidance
  runs `torch.autograd.grad` inside the denoising loop, which cannot be captured
  by CUDA graphs — so a checkpoint trained with `compile_mode='reduce-overhead'`
  crashes (`cudaErrorStreamCaptureInvalidated`) if compiled under RTC. Pass
  `--compile` to opt in; it is forced to a non-CUDA-graph mode (`default`).
  The server still does a startup warmup pass regardless.
