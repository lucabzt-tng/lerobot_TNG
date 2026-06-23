"""
Async inference client for Counter-Strike + pi05 policy.

Replaces rtc_inference.py (synchronous chunk-then-execute) with the
gRPC async inference stack so inference and action execution overlap,
eliminating idle frames between chunks.

Usage
-----
# Terminal 1 — start the policy server (from lerobot_TNG root):
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080

# Terminal 2 — run the client:
cd src   # so lerobot is importable, or activate the uv env
python inference/counterstrike/async_inference.py \
    --model_path ~/bozzetti/models/counterstrike/lerobot_pi05_test/050000/pretrained_model \
    --server_address 127.0.0.1:8080

Keyboard controls (same as rtc_inference.py):
  x  — pause / resume
  q  — quit (while paused)
"""

import argparse
import pickle
import subprocess
import sys
import threading
import time
from queue import Queue

import cv2
import grpc
import numpy as np
import pygame
import vgamepad as vg
import Xlib
import Xlib.display
from pynput import keyboard

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

# ---------------------------------------------------------------------------
# Policy / observation constants (must match config.json)
# ---------------------------------------------------------------------------

STATES = [
    "Left Stick X", "Left Stick Y", "Left Trigger",
    "Right Stick X", "Right Stick Y", "Right Trigger",
    "A", "B", "X", "Y",
    "LB (Left Bumper)", "RB (Right Bumper)",
    "Mini map large", "Start", "unknown",
    "Left Stick (press)", "Right Stick (press)", "Screenshot",
    "Dpad Left", "Dpad Right", "Dpad Down", "Dpad Up",
]

AXES_DIM = 6
BUTTONS_DIM = 16
ACTION_DIM = 22       # output_features.action.shape[0]
IMAGE_SIZE = 512      # input_features.observation.images._view.shape[1]

# Feature metadata the server uses to unpack raw observation dicts.
# State: build_dataset_frame looks up each name as an individual key in the raw obs dict.
# Image: build_dataset_frame looks for raw_obs["_view"] (strips the "observation.images." prefix).
LEROBOT_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": STATES,
    },
    "observation.images._view": {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),  # HWC — server permutes to CHW internally
        "names": ["height", "width", "channels"],
    },
}

DEATHMATCH_PROMPT = (
    "You are playing counterstrike. You are a counter-terrorist and playing team deathmatch. "
    "Search and shoot the enemies."
)
PROMPT = DEATHMATCH_PROMPT

FPS = 60
FRAME_DURATION = 1.0 / FPS
ACTIONS_PER_CHUNK = 50   # matches config.json chunk_size
CHUNK_SIZE_THRESHOLD = 0.5  # send new obs when queue ≤ 50 % full

# Resting gamepad state matching the training data recording conventions.
NO_MOVEMENT_AXES = np.array([0, 0, 0, 0, 0, -1], dtype=np.float32)
NO_MOVEMENT_BUTTONS = np.zeros(BUTTONS_DIM, dtype=np.float32)

# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

class FFmpegX11Grabber:
    """X11 region capture via ffmpeg x11grab → rawvideo (bgr0) pipe."""

    def __init__(self, x: int, y: int, w: int, h: int, fps: int = 60, display: str = ":1"):
        self.x, self.y, self.w, self.h, self.fps = x, y, w, h, fps
        self.frame_size = w * h * 4  # bgr0 = 4 bytes/pixel
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._frame = np.empty((h, w, 4), dtype=np.uint8)
        self._ts = None
        self._seq = 0
        self._stop = False

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "x11grab", "-framerate", str(fps),
            "-video_size", f"{w}x{h}", "-i", f"{display}+{x},{y}",
            "-pix_fmt", "bgr0", "-f", "rawvideo", "pipe:1",
        ]
        self._cmd = cmd
        self._p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8, stderr=subprocess.DEVNULL)
        self._t = threading.Thread(target=self._reader_loop, daemon=True)
        self._t.start()

    def _reader_loop(self):
        try:
            out = self._p.stdout
            while not self._stop:
                buf = out.read(self.frame_size)
                if len(buf) != self.frame_size:
                    break
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((self.h, self.w, 4))
                now = time.perf_counter()
                with self._cv:
                    np.copyto(self._frame, arr)
                    self._ts = now
                    self._seq += 1
                    self._cv.notify_all()
        finally:
            self._stop = True

    def grab_latest(self):
        """Returns (frame_copy, ts, seq) without blocking, or (None, None, None)."""
        with self._lock:
            if self._ts is None:
                return None, None, None
            return self._frame.copy(), self._ts, self._seq

    def grab_next(self, last_seq=None, timeout: float = 0.25):
        """Blocks until a new frame arrives. Returns (frame_copy, ts, seq)."""
        end = time.perf_counter() + timeout
        with self._cv:
            target = self._seq if last_seq is None else last_seq
            while self._seq <= target:
                remaining = end - time.perf_counter()
                if remaining <= 0:
                    return None, None, None
                self._cv.wait(timeout=remaining)
            return self._frame.copy(), self._ts, self._seq

    def close(self):
        self._stop = True
        try:
            self._p.terminate()
        except Exception:
            pass
        try:
            self._p.stdout.close()
        except Exception:
            pass
        self._t.join(timeout=1.0)
        try:
            self._p.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._p.kill()
            self._p.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numpy_flip(im: np.ndarray) -> np.ndarray:
    """Convert bgr0 frame → RGB uint8."""
    return np.flip(np.asarray(im, dtype=np.uint8)[:, :, :3], 2)


def get_window_dims() -> dict:
    disp = Xlib.display.Display()
    window = disp.get_input_focus().focus
    geometry = window.get_geometry()
    parent = window.query_tree().parent
    pg = parent.get_geometry()
    return {"left": pg.x + geometry.x, "top": pg.y + geometry.y,
            "width": geometry.width, "height": geometry.height}


def clamp11(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def execute_actions(gamepad, axes: np.ndarray, buttons: np.ndarray):
    stick_scale = 32767
    trigger_scale = 255

    gamepad.left_joystick(
        x_value=int(clamp11(axes[0]) * stick_scale),
        y_value=int(clamp11(axes[1]) * stick_scale),
    )
    gamepad.right_joystick(
        x_value=int(clamp11(axes[3]) * stick_scale),
        y_value=int(clamp11(axes[4]) * stick_scale),
    )
    gamepad.left_trigger(value=max(0, int(clamp11(axes[2]) * trigger_scale)))
    gamepad.right_trigger(value=max(0, int(clamp11(axes[5]) * trigger_scale)))

    btn_map = [
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_A,               0),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_B,               1),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_X,               2),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,               3),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,   4),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,  5),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,            6),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_START,           7),
        # index 8 ("unknown") skipped intentionally
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,      9),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,     10),
        (vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE,           11),
    ]
    for btn, idx in btn_map:
        if buttons[idx] > 0.5:
            gamepad.press_button(btn)
        else:
            gamepad.release_button(btn)

    if len(buttons) > 12:
        dpad_map = [
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,  12),
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT, 13),
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,  14),
            (vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,    15),
        ]
        for btn, idx in dpad_map:
            if buttons[idx] > 0.5:
                gamepad.press_button(btn)
            else:
                gamepad.release_button(btn)

    gamepad.update()


def build_raw_obs(axes: np.ndarray, buttons: np.ndarray, image_hwc: np.ndarray, task: str) -> dict:
    """
    Build a raw observation dict compatible with LEROBOT_FEATURES.

    build_dataset_frame expects individual named floats for the state
    (one key per STATES entry) and the image under the bare camera name "_view".
    """
    obs = {}
    for i, name in enumerate(STATES[:AXES_DIM]):
        obs[name] = float(axes[i])
    for i, name in enumerate(STATES[AXES_DIM:]):
        obs[name] = float(buttons[i])
    obs["_view"] = image_hwc   # (H, W, C) uint8 — server resizes + permutes to CHW
    obs["task"] = task
    return obs


# ---------------------------------------------------------------------------
# gRPC async client (no Robot instance needed)
# ---------------------------------------------------------------------------

class DirectAsyncClient:
    """
    Thin gRPC wrapper around PolicyServer for use without a robot.

    Handles handshake, observation sending, async action reception, and the
    action queue / chunk-size-threshold logic from RobotClient — but builds
    observations from the CS gamepad + screen capture instead of robot.motor.
    """

    def __init__(
        self,
        server_address: str,
        model_path: str,
        policy_device: str = "cuda",
        actions_per_chunk: int = ACTIONS_PER_CHUNK,
        chunk_size_threshold: float = CHUNK_SIZE_THRESHOLD,
    ):
        self._chunk_size_threshold = chunk_size_threshold
        self.actions_per_chunk = actions_per_chunk

        self.channel = grpc.insecure_channel(server_address, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.policy_config = RemotePolicyConfig(
            policy_type="pi05",
            pretrained_name_or_path=model_path,
            lerobot_features=LEROBOT_FEATURES,
            actions_per_chunk=actions_per_chunk,
            device=policy_device,
        )

        self.action_queue: Queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_chunk_size: int = -1

        self._latest_ts_lock = threading.Lock()
        self._latest_timestep: int = 0

        self.shutdown_event = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()  # first observation always forced through

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self) -> bool:
        """Handshake + load policy on server. Returns True on success."""
        try:
            self.stub.Ready(services_pb2.Empty())
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))
            )
            self.shutdown_event.clear()
            print(f"[client] Connected. Policy '{self.policy_config.policy_type}' loaded on server.")
            return True
        except grpc.RpcError as e:
            print(f"[client] Connection failed: {e}")
            return False

    def send_observation(self, raw_obs: dict, timestep: int, must_go: bool = False) -> bool:
        timed_obs = TimedObservation(
            timestamp=time.time(),
            observation=raw_obs,
            timestep=timestep,
            must_go=must_go,
        )
        try:
            obs_iter = send_bytes_in_chunks(
                pickle.dumps(timed_obs), services_pb2.Observation, silent=True
            )
            self.stub.SendObservations(obs_iter)
            return True
        except grpc.RpcError as e:
            print(f"[client] Send observation error: {e}")
            return False

    def receive_actions(self):
        """Background thread: continuously polls GetActions and merges into queue."""
        while self.running:
            try:
                resp = self.stub.GetActions(services_pb2.Empty())
                if not resp.data:
                    continue
                timed_actions: list[TimedAction] = pickle.loads(resp.data)
                if not timed_actions:
                    continue

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._merge_incoming_actions(timed_actions)
                self.must_go.set()  # next time queue empties → force a must-go obs

            except grpc.RpcError as e:
                if self.running:
                    print(f"[client] Receive actions error: {e}")
                    time.sleep(0.05)

    def _merge_incoming_actions(self, incoming: list[TimedAction]):
        """
        Blend incoming chunk into the queue on overlapping timesteps
        (weighted_average: 30 % old + 70 % new), mirroring RobotClient behaviour.
        """
        with self._latest_ts_lock:
            latest = self._latest_timestep

        with self.action_queue_lock:
            existing = {a.get_timestep(): a.get_action() for a in self.action_queue.queue}
            merged: Queue = Queue()
            for new_a in incoming:
                ts = new_a.get_timestep()
                if ts <= latest:
                    continue
                if ts in existing:
                    blended = 0.3 * existing[ts] + 0.7 * new_a.get_action()
                    merged.put(TimedAction(
                        timestamp=new_a.get_timestamp(), timestep=ts, action=blended
                    ))
                else:
                    merged.put(new_a)
            self.action_queue = merged

    def pop_action(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Pop the next action. Returns (axes[6], buttons[16]) or None if queue empty."""
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None
            timed_action = self.action_queue.get_nowait()

        with self._latest_ts_lock:
            self._latest_timestep = timed_action.get_timestep()

        action = timed_action.get_action().cpu().numpy()  # shape (22,)
        return action[:AXES_DIM], action[AXES_DIM:AXES_DIM + BUTTONS_DIM]

    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def ready_to_send_observation(self) -> bool:
        """True when the queue has dropped to ≤ chunk_size_threshold of max chunk size."""
        if self.action_chunk_size <= 0:
            return True  # send freely until the first chunk arrives
        with self.action_queue_lock:
            qsize = self.action_queue.qsize()
        return qsize / self.action_chunk_size <= self._chunk_size_threshold

    def stop(self):
        self.shutdown_event.set()
        self.channel.close()


# ---------------------------------------------------------------------------
# Keyboard handlers (global state — same pattern as rtc_inference.py)
# ---------------------------------------------------------------------------

pause_event = threading.Event()
pause_event.set()
paused = False
running = True
gamepad = None
no_movement_axes = NO_MOVEMENT_AXES.copy()
no_movement_buttons = NO_MOVEMENT_BUTTONS.copy()


def _on_press_start(key):
    try:
        if hasattr(key, "char") and key.char == "x":
            return False
    except AttributeError:
        pass


def _on_press_control(key):
    global paused, running
    try:
        ch = key.char
    except AttributeError:
        return
    if ch == "x" and not paused:
        execute_actions(gamepad, no_movement_axes, no_movement_buttons)
        time.sleep(0.5)
        print("Paused. Press 'x' to resume or 'q' to quit.")
        paused = True
        pause_event.clear()
    elif ch == "x" and paused:
        print("Resumed.")
        pause_event.set()
        paused = False
    elif ch == "q" and paused:
        print("Quitting.")
        pause_event.set()
        running = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    server_address: str,
    model_path: str,
    policy_device: str,
    actions_per_chunk: int,
    chunk_size_threshold: float,
):
    global gamepad, no_movement_axes, no_movement_buttons

    print("Press 'x' to start.")
    with keyboard.Listener(on_press=_on_press_start) as listener:
        listener.join()
    print("Starting async inference...")

    dims = get_window_dims()
    cap = FFmpegX11Grabber(
        x=dims["left"], y=dims["top"], w=dims["width"], h=dims["height"],
        fps=60, display=":1",
    )

    gamepad = vg.VX360Gamepad()
    time.sleep(3)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No virtual gamepad detected. Exiting.")
        pygame.quit()
        cap.close()
        sys.exit(1)
    pygame.joystick.Joystick(0).init()

    execute_actions(gamepad, no_movement_axes, no_movement_buttons)

    # Wait for the first frame before connecting to the server
    frame, _, _ = cap.grab_next(last_seq=None, timeout=2.0)
    if frame is None:
        print("No frame received from screen capture. Exiting.")
        cap.close()
        sys.exit(1)
    resized_frame = cv2.resize(numpy_flip(frame), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    client = DirectAsyncClient(
        server_address=server_address,
        model_path=model_path,
        policy_device=policy_device,
        actions_per_chunk=actions_per_chunk,
        chunk_size_threshold=chunk_size_threshold,
    )
    if not client.start():
        cap.close()
        sys.exit(1)

    receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    receiver_thread.start()

    # Kick off inference with the first observation (must_go so the server processes it immediately)
    raw_obs = build_raw_obs(no_movement_axes, no_movement_buttons, resized_frame, PROMPT)
    client.send_observation(raw_obs, timestep=0, must_go=True)
    print("Initial observation sent. Waiting for first action chunk...")

    last_axes = no_movement_axes.copy()
    last_buttons = no_movement_buttons.copy()
    timestep = 0
    last_frame_time = time.perf_counter()
    test_image_saved = False
    iter_count = 0

    listener_control = keyboard.Listener(on_press=_on_press_control)
    listener_control.start()

    try:
        while running:
            if paused:
                execute_actions(gamepad, no_movement_axes, no_movement_buttons)
            pause_event.wait()

            now = time.perf_counter()
            if now - last_frame_time < FRAME_DURATION:
                time.sleep(0.001)
                continue

            last_frame_time = now
            iter_count += 1

            # Capture the latest frame
            frame, _, _ = cap.grab_latest()
            if frame is not None:
                resized_frame = cv2.resize(
                    numpy_flip(frame), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
                )

            if iter_count % 120 == 0 and not test_image_saved:
                cv2.imwrite("test_image_async.png", cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
                test_image_saved = True

            # Execute next action; hold position if queue is still empty
            result = client.pop_action()
            if result is not None:
                last_axes, last_buttons = result
                execute_actions(gamepad, last_axes, last_buttons)
                timestep += 1
            else:
                execute_actions(gamepad, no_movement_axes, no_movement_buttons)

            # Send a new observation whenever the queue is running low
            if client.ready_to_send_observation():
                with client.action_queue_lock:
                    queue_empty = client.action_queue.empty()
                must_go = client.must_go.is_set() and queue_empty
                if must_go:
                    client.must_go.clear()
                raw_obs = build_raw_obs(last_axes, last_buttons, resized_frame, PROMPT)
                client.send_observation(raw_obs, timestep=timestep, must_go=must_go)

    except KeyboardInterrupt:
        pass
    finally:
        listener_control.stop()
        execute_actions(gamepad, no_movement_axes, no_movement_buttons)
        client.stop()
        cap.close()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained pi05 checkpoint directory.")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080",
                        help="host:port of the running policy server.")
    parser.add_argument("--policy_device", type=str, default="cuda",
                        help="Device for policy inference on the server (cuda / cpu).")
    parser.add_argument("--actions_per_chunk", type=int, default=ACTIONS_PER_CHUNK,
                        help="Max actions to request per inference call (≤ config chunk_size).")
    parser.add_argument("--chunk_size_threshold", type=float, default=CHUNK_SIZE_THRESHOLD,
                        help="Send a new observation when queue drops to this fraction of chunk size.")
    args = parser.parse_args()
    main(
        server_address=args.server_address,
        model_path=args.model_path,
        policy_device=args.policy_device,
        actions_per_chunk=args.actions_per_chunk,
        chunk_size_threshold=args.chunk_size_threshold,
    )
