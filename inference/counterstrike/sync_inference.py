### For a single bot to test Gr00t :

### bot_kick
### sv_cheats 1
### bot_add
###Freeze bot and allow not too shot:
### bot_stop 1
### bot_dont_shoot 1

### look at the place where to put bot and do:

### bot_place

import numpy as np
import cv2
import argparse

import sys

import Xlib
import Xlib.display

from inference_client import RTCInferenceClient

import time

import vgamepad as vg

from pynput import keyboard

import pygame

import subprocess

from collections import deque

import threading

pause_event = threading.Event()
pause_event.set()  # Initially not paused

paused = False

running = True


class FFmpegX11Grabber:
    """
    X11 region capture via ffmpeg x11grab -> rawvideo (bgr0) pipe.
    Continuously reads frames in a background thread and keeps only the latest.
    grab_latest() returns immediately.
    grab_next() waits for the next frame (useful for sync points).
    """

    def __init__(self, x: int, y: int, w: int, h: int, fps: int = 60, display=":0.0"):
        self.x, self.y, self.w, self.h, self.fps = x, y, w, h, fps
        self.display = display

        self.frame_size = w * h * 4  # bgr0 = 4 bytes/pixel
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._frame = np.empty((h, w, 4), dtype=np.uint8)
        self._ts = None
        self._seq = 0
        self._stop = False

        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",

            # capture from X11
            "-f", "x11grab",
            "-framerate", str(fps),
            "-video_size", f"{w}x{h}",
            "-i", f"{display}+{x},{y}",

            # output raw frames
            "-pix_fmt", "bgr0",
            "-f", "rawvideo",
            "pipe:1",
        ]

        self._cmd = cmd

        # Use a reasonably large pipe buffer
        self._p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8, stderr=subprocess.DEVNULL)

        self._t = threading.Thread(target=self._reader_loop)
        self._t.start()

    def _reader_loop(self):
        try:
            out = self._p.stdout
            while not self._stop:
                buf = out.read(self.frame_size)
                if len(buf) != self.frame_size:
                    break  # ffmpeg ended or pipe broke

                # Copy bytes into preallocated numpy array (no per-frame allocation)
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
        """Immediate: returns (frame, ts, seq) or (None, None, None) if not ready yet."""
        with self._lock:
            if self._ts is None:
                return None, None, None
            return self._frame, self._ts, self._seq

    def grab_next(self, last_seq=None, timeout=0.25):
        """
        Wait until a newer frame than last_seq arrives.
        Returns (frame, ts, seq) or (None, None, None) on timeout.
        """
        end = time.perf_counter() + timeout
        with self._cv:
            target = self._seq if last_seq is None else last_seq
            while self._seq <= target:
                remaining = end - time.perf_counter()
                if remaining <= 0:
                    return None, None, None
                self._cv.wait(timeout=remaining)
            return self._frame, self._ts, self._seq

    def reset_ffmpeg_process(self):
        """
        Stops and restarts the FFmpeg process to clear the internal buffer.
        """
        print("Resetting FFmpeg process...")

        # Stop the current FFmpeg process
        self._stop = True
        try:
            if self._p.poll() is None:
                self._p.terminate()
        except Exception:
            pass

        try:
            if self._p.stdout:
                self._p.stdout.close()
        except Exception:
            pass
        self._t.join(timeout=1.0)
        try:
            self._p.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._p.kill()
            self._p.wait()

        # Restart the process
        self._stop = False
        self._p = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1024)
        self._t = threading.Thread(target=self._reader_loop)
        self._t.start()

        print("FFmpeg process reset successfully.")

    def close(self):
        self._stop = True
        try:
            if self._p.poll() is None:
                self._p.terminate()
        except Exception:
            pass

        try:
            if self._p.stdout:
                self._p.stdout.close()
        except Exception:
            pass
        self._t.join(timeout=1.0)
        try:
            self._p.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._p.kill()
            self._p.wait()


### returns the image in RGB format
def numpy_flip(im):
    """ Most efficient Numpy version as of now. """
    frame = np.array(im, dtype=np.uint8)
    return np.flip(frame[:, :, :3], 2)


def getCurrentWindowDimensions_clean():
    # Open the X display
    disp = Xlib.display.Display()

    # Get the currently focused window
    window = disp.get_input_focus().focus

    # Query the geometry of the focused window (client window)
    ## is returned relative to parent, so I have to add parent geometry
    geometry = window.get_geometry()

    # Get the parent window (as it could be in a container)
    parent = window.query_tree().parent

    # Query the position of the parent window relative to the root window
    parent_geometry = parent.get_geometry()

    # Calculate absolute position of the window on the screen
    abs_x = parent_geometry.x + geometry.x
    abs_y = parent_geometry.y + geometry.y

    return {"left": abs_x, "top": abs_y, "width": geometry.width, "height": geometry.height}


# Axis mapping (for Xbox controller)
AXIS_MAP = {
    0: "Left Stick X",  # Left Stick X-axis
    1: "Left Stick Y",  # Left Stick Y-axis
    2: "Left Trigger",  # Left Trigger
    3: "Right Stick X",  # Right Stick X-axis
    4: "Right Stick Y",  # Right Stick Y-axis
    5: "Right Trigger"  # Right Trigger
}

# Button mapping (for Xbox controller)
BUTTON_MAP = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    4: "LB (Left Bumper)",
    5: "RB (Right Bumper)",
    6: "Mini map large",
    7: "Start",
    8: "unknown",
    9: "Left Stick (press)",
    10: "Right Stick (press)",
    11: "Screenshot",
    12: "dpad left",
    13: "dpad right",
    14: "dpad down",
    15: "dpad up"
    # Buttons 11-13 are typically unused, but you can add them if you need
}

STATES = [
    "Left Stick X",  # Left Stick X-axis
    "Left Stick Y",  # Left Stick Y-axis
    "Left Trigger",  # Left Trigger
    "Right Stick X",  # Right Stick X-axis
    "Right Stick Y",  # Right Stick Y-axis
    "Right Trigger",
    "A",
    "B",
    "X",
    "Y",
    "LB (Left Bumper)",
    "RB (Right Bumper)",
    "Mini map large",
    "Start",
    "unknown",
    "Left Stick (press)",
    "Right Stick (press)",
    "Screenshot",
    "Dpad Left",
    "Dpad Right",
    "Dpad Down",
    "Dpad Up"
]

ARMS_RACE_PROMPT = "You are playing counterstrike. You are a counter-terrorist and playing Arms Race. Friends are marked with blue text above their heads. Search and shoot the enemies."
DEATHMATCH_PROMPT = "You are playing counterstrike. You are a counter-terrorist and playing team deathmatch. Search and shoot the enemies."
PROMPT = DEATHMATCH_PROMPT

FPS = 60

FRAME_DURATION = 1 / FPS

ACTION_HORIZON_USED = 16

DIMS_VIDEO_RESIZED = {
    "height": 512,
    "width": 512,
}

VIDEO_PATH = "cs_gr00t_test_vid.mp4"

IMAGE_GRAB_DURATION = 6e-3

# OBSERVATION_INDICES = [-20 , -10, 0]

OBSERVATION_INDICES = [0]

SHIFTED_OBSERVATION_INDICES = [i - 1 for i in OBSERVATION_INDICES]

print("obs indices used: ", OBSERVATION_INDICES)

print("shifted obs indices used: ", SHIFTED_OBSERVATION_INDICES)

OBSERVATION_HORIZON_LEN = abs(max(SHIFTED_OBSERVATION_INDICES, key=abs))

print("observation horizon len: ", OBSERVATION_HORIZON_LEN)

OBSERVATION_INDICES = SHIFTED_OBSERVATION_INDICES

DATA_CONFIG_FULL = True


# cv_writer_global.write(cv2_frame)


### important: Have to set:               xinput set-prop "Logitech USB Receiver Mouse" "libinput Accel Profile Enabled" 0 1
### in order to disable mouse acceleration
### important: inference time is about 90ms sometimes a bit less. This delay is not present when recording, so it has to be taken into account.


def numpy_flip(im):
    """ Most efficient Numpy version as of now. """
    frame = np.array(im, dtype=np.uint8)
    return np.flip(frame[:, :, :3], 2)


def push_away_from_deadzone(x, deadzone=0.07, threshold=0.01):
    if x == 0:
        return 0.0

    elif abs(x) < threshold:

        return 0.0

    return (deadzone + (1 - deadzone) * abs(x)) * (1 if x > 0 else -1)


def push_away_from_deadzone_array(x, deadzone=0.07):
    x = np.asarray(x)

    sign = np.sign(x)
    abs_x = np.abs(x)

    # Apply mapping everywhere
    y = sign * (deadzone + (1 - deadzone) * abs_x)

    # Preserve exact zeros
    y[x == 0] = 0.0

    return y


def clamp_between_minus1_and_1_array(x):
    return np.clip(x, -1.0, 1.0)


def execute_actions(gamepad, axes: np.array, buttons: np.array):
    y_axis_inv_correction_factor = 1.0

    stick_scale = 32767
    trigger_scale = 255

    # print("vals given to gamepad: axes: ", int(axes[0] * stick_scale), int(axes[1] * stick_scale), int(axes[3] * stick_scale), int(axes[4] * stick_scale), int(axes[2] * trigger_scale), int(axes[5] * trigger_scale))

    gamepad.left_joystick(x_value=int((clamp_value_between_m_1_and_1(axes[0])) * stick_scale), y_value=int(
        (clamp_value_between_m_1_and_1(axes[1])) * stick_scale * y_axis_inv_correction_factor))
    gamepad.right_joystick(x_value=int((clamp_value_between_m_1_and_1(axes[3])) * stick_scale), y_value=int(
        (clamp_value_between_m_1_and_1(axes[4])) * stick_scale * y_axis_inv_correction_factor))
    # Triggers map model output [-1, 1] (-1 = released) to vgamepad's UNSIGNED byte
    # [0, 255]. A negative value must clamp to 0 (released): passing it through would
    # wrap modulo 256 into a positive byte and falsely pull the trigger (e.g. -0.85 *
    # 255 = -216 -> 40 -> ~16% pull -> constant firing).
    gamepad.left_trigger(value=max(0, int(clamp_value_between_m_1_and_1(axes[2]) * trigger_scale)))
    gamepad.right_trigger(value=max(0, int(clamp_value_between_m_1_and_1(axes[5]) * trigger_scale)))

    if buttons[0] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    if buttons[1] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    if buttons[2] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

    if buttons[3] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)

    if buttons[4] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)

    if buttons[5] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

    if buttons[6] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK)

    if buttons[7] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_START)

    # if buttons[8] > 0.5:
    #   gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
    # else:
    #   gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)

    if buttons[9] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)

    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)

    if buttons[10] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB)

    if buttons[11] > 0.5:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE)

    if len(buttons) > 12:

        if buttons[12] > 0.5:
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
        else:
            gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)

        if buttons[13] > 0.5:
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)
        else:
            gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT)

        if buttons[14] > 0.5:
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
        else:
            gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)

        if buttons[15] > 0.5:
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
        else:
            gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)

    gamepad.update()


def on_press_to_stop_or_resume(key):
    global paused

    global running

    global gamepad

    global no_movement_axes

    global no_movement_buttons

    try:
        if key.char == 'x' and not paused:
            execute_actions(gamepad, no_movement_axes, no_movement_buttons)
            time.sleep(0.5)
            print("Paused. Press 'x' to resume or 'q' to quit.")
            paused = True
            pause_event.clear()  # block loop

        elif key.char == 'x' and paused:
            print("Resumed")
            pause_event.set()  # unblock loop
            paused = False
        elif key.char == 'q' and paused:
            print("Quitting")
            pause_event.set()  # in case it's paused
            running = False
    except AttributeError:
        pass


def on_press(key):
    try:
        if hasattr(key, 'char') and key.char == 'x':
            # Stop the listener
            return False
    except AttributeError:
        pass


def clamp_value_between_m_1_and_1(value):
    return max(-1, min(1, value))


class FrameMemory:
    def __init__(self, max_size=40):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)  # deque will automatically remove oldest when limit is exceeded

    def add(self, frame, observation_axes, observation_buttons):
        # Add new frame and observation at the most recent end
        self.memory.append((frame, observation_axes, observation_buttons))

    def get_all(self):
        # Return all stored frames and observations
        return list(self.memory)

    def get_recent(self):
        # Return the most recent frame and observation
        return self.memory[-1] if self.memory else None

    def get_oldest(self):
        # Return the oldest frame and observation
        return self.memory[0] if self.memory else None

    def get_by_indices(self, indices):
        # Extract frames and observations based on indices
        selected_frames = []
        selected_observations_axes = []
        selected_observations_buttons = []

        for i in indices:
            frame, observation_axes, observation_buttons = self.memory[i]
            selected_frames.append(frame)
            selected_observations_axes.append(observation_axes)
            selected_observations_buttons.append(observation_buttons)

        # Convert to np.array: Assuming frames are NumPy arrays or similar structured data
        frames_array = np.array(selected_frames)
        observations_axes_array = np.array(selected_observations_axes)
        observations_buttons_array = np.array(selected_observations_buttons)

        return frames_array, observations_axes_array, observations_buttons_array


def main(server_host: str, server_port: int):
    print("Press x to let the policy play.")

    # Start listening to the keyboard
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    print("Key pressed. Gr00t starts playing now.")

    DIMS_VIDEO = getCurrentWindowDimensions_clean()

    ### init video writer
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{DIMS_VIDEO_RESIZED['width']}x{DIMS_VIDEO_RESIZED['height']}",
        "-r", str(FPS / 4),
        "-i", "-",  # read from stdin
        "-c:v", "libx264",  # NVIDIA hardware encoder
        "-rc", "constqp",
        "-qp", "18",
        "-preset", "fast",
        VIDEO_PATH
    ]

    cap = FFmpegX11Grabber(x=DIMS_VIDEO['left'], y=DIMS_VIDEO['top'], w=DIMS_VIDEO['width'], h=DIMS_VIDEO['height'],
                           fps=60, display=":1")

    global gamepad

    gamepad = vg.VX360Gamepad()

    time.sleep(3)  # TODO why sleep 3 seconds?

    #### debug input to controller

    pygame.init()

    # Initialize joystick
    pygame.joystick.init()

    # Check if there is at least one joystick connected
    if pygame.joystick.get_count() == 0:
        print("No virtual gamepad detected! Evtl wait a bit longer for init of gamepad. That takes some time")
        pygame.quit()
        exit()

    # Get the first joystick (index 0)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    global no_movement_axes

    global no_movement_buttons

    no_movement_axes = np.array([0, 0, 0, 0, 0, -1], dtype=np.float32)

    small_movement_axes = np.array([0.1, 0, 0, 0.0, 0, 0], dtype=np.float32)

    team_select_axes = np.array([0, 0, 0, 0.07, 0, 0], dtype=np.float32)

    if DATA_CONFIG_FULL == True:

        no_movement_buttons = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    else:

        no_movement_buttons = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    execute_actions(gamepad, no_movement_axes, no_movement_buttons)

    frame, ts, seq = cap.grab_next(last_seq=None, timeout=0.5)

    img = np.array(frame)

    img = numpy_flip(img)

    img = img.astype(np.uint8)

    resized_frame = cv2.resize(img, (DIMS_VIDEO_RESIZED['width'], DIMS_VIDEO_RESIZED['height']),
                               interpolation=cv2.INTER_CUBIC)

    if len(OBSERVATION_INDICES) > 1:
        frame_memory = FrameMemory(max_size=OBSERVATION_HORIZON_LEN + 1)

        for _ in range(OBSERVATION_HORIZON_LEN):
            frame_memory.add(resized_frame, no_movement_axes, no_movement_buttons)

    client = RTCInferenceClient(host=server_host, port=server_port)

    # Synchronous open-loop chunking: fetch a full action chunk, execute its first
    # ACTION_HORIZON_USED actions one per frame, hold the controller at neutral while
    # the next chunk is queried, then repeat. No RTC blending happens here -- see
    # rtc_inference.py for the RTC (one-blended-action-per-frame) path.
    print(f"Connected to sync inference server; using {ACTION_HORIZON_USED} actions per chunk.")

    # Track the last executed action so it can be fed back as state on the next query.
    last_axes = no_movement_axes
    last_buttons = no_movement_buttons

    # Current chunk and how many of its actions we've executed. chunk_axes=None forces
    # a fetch on the first frame.
    chunk_axes = None
    chunk_buttons = None
    chunk_idx = 0

    iter = 0

    start_time = time.perf_counter()

    last_frame_time = start_time

    current_time = time.perf_counter()

    test_image_saved = False

    listener_stop_resume = keyboard.Listener(on_press=on_press_to_stop_or_resume)
    listener_stop_resume.start()

    while running:

        if paused:
            execute_actions(gamepad, no_movement_axes, no_movement_buttons)

        pause_event.wait()  # Wait here if paused

        current_time = time.perf_counter()
        elapsed_time = abs(current_time - last_frame_time)

        if (elapsed_time >= FRAME_DURATION):

            iter = iter + 1

            frame, ts, seq = cap.grab_latest()

            img = np.array(frame)

            img = numpy_flip(img)

            img = img.astype(np.uint8)

            last_frame_time = current_time

            resized_frame = cv2.resize(img, (DIMS_VIDEO_RESIZED['width'], DIMS_VIDEO_RESIZED['height']),
                                       interpolation=cv2.INTER_CUBIC)

            if iter % 60 == 0:
                if not test_image_saved:
                    cv2.imwrite('test_image.png', resized_frame)
                    test_image_saved = True

            # Re-query only when the current chunk's first ACTION_HORIZON_USED actions
            # are exhausted (or on the first frame). While the blocking chunk request is
            # in flight, hold the controller at neutral so it does nothing meanwhile.
            if chunk_axes is None or chunk_idx >= min(ACTION_HORIZON_USED, len(chunk_axes)):
                execute_actions(gamepad, no_movement_axes, no_movement_buttons)

                observation_dict = {
                    "state.axes": np.expand_dims(np.array(last_axes), axis=0).astype(np.float32),
                    "state.buttons": np.expand_dims(np.array(last_buttons), axis=0).astype(np.float32),
                    "video._view": np.expand_dims(resized_frame, axis=0).astype(np.uint8),
                    "annotation.human.task_description": [PROMPT]
                }

                if len(OBSERVATION_INDICES) > 1:
                    frames_array, observations_axes_array, observations_buttons_array = frame_memory.get_by_indices(
                        OBSERVATION_INDICES)
                    observation_dict = {
                        "state.axes": np.expand_dims(np.array(observations_axes_array), axis=0).astype(np.float32),
                        "state.buttons": np.expand_dims(np.array(observations_buttons_array), axis=0).astype(np.float32),
                        "video._view": np.expand_dims(frames_array, axis=0).astype(np.uint8),
                        "annotation.human.task_description": [PROMPT]
                    }

                action_chunk = client.get_action_chunk(observation_dict)
                chunk_axes = action_chunk["action.axes"]  # (chunk_size, 6)
                chunk_buttons = action_chunk["action.buttons"]  # (chunk_size, num_buttons)
                chunk_idx = 0
                print(f"New action chunk obtained: {chunk_axes.shape[0]} actions "
                      f"(using first {min(ACTION_HORIZON_USED, len(chunk_axes))}).")

            # Execute the next action from the current chunk.
            current_axes = chunk_axes[chunk_idx]
            current_buttons = chunk_buttons[chunk_idx]
            chunk_idx += 1

            execute_actions(gamepad, current_axes, current_buttons)

            if len(OBSERVATION_INDICES) > 1:
                frame_memory.add(resized_frame, current_axes, current_buttons)

            last_axes = current_axes
            last_buttons = current_buttons

    else:
        listener_stop_resume.stop()
        cap.close()
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run synchronous (open-loop chunk) inference client against the inference server.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Inference server host (use 127.0.0.1 with SSH tunnel).")
    parser.add_argument("--port", type=int, default=5555, help="Inference server port.")
    args = parser.parse_args()

    main(server_host=args.host, server_port=args.port)