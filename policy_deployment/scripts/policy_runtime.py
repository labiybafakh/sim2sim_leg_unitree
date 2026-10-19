import json
import math
from pathlib import Path

import numpy as np


def finite(values, shape):
    result = np.asarray(values, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite values with shape {shape}")
    return result


def load_config(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate config key: {key}")
            result[key] = value
        return result

    path = Path(path).resolve()
    config = json.loads(path.read_text(), object_pairs_hook=unique_object)
    directory = config["policy_directory"]
    if not isinstance(directory, str) or not directory:
        raise ValueError("policy_directory must be a nonempty path")
    config["policy_directory"] = str((path.parent / directory).resolve())
    for key in ("action_scale", "policy_frequency_hz"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    joints = config["joints"]
    if len(joints) != 12:
        raise ValueError("Expected exactly 12 configured joints")
    names = [joint["name"] for joint in joints]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 12:
        raise ValueError("Joint names must be nonempty and unique")
    for key in ("default_position", "stiffness", "damping"):
        values = finite([joint[key] for joint in joints], (12,))
        if key != "default_position" and (values < 0).any():
            raise ValueError("Joint gains must be nonnegative")
    joy = config["joystick"]
    if len(joy["axes"]) != 3 or any(type(i) is not int or i < 0 for i in joy["axes"]):
        raise ValueError("joystick.axes must contain three nonnegative integers")
    finite(joy["scales"], (3,))
    if not math.isfinite(joy["deadzone"]) or not 0 <= joy["deadzone"] < 1:
        raise ValueError("Joystick deadzone must be in [0, 1)")
    buttons = [joy["buttons"][key] for key in ("start", "stop", "l1", "r1")]
    if any(type(i) is not int or i < 0 for i in buttons) or len(set(buttons)) != 4:
        raise ValueError("Joystick buttons must be four distinct nonnegative integers")
    return config


class Policy:
    def __init__(self, config, model_path=None, threads=1):
        import torch

        if threads < 1:
            raise ValueError("inference_threads must be positive")
        self.torch = torch
        self.config = config
        self.defaults = finite([j["default_position"] for j in config["joints"]], (12,))
        torch.set_num_threads(threads)
        path = Path(model_path) if model_path else Path(config["policy_directory"]) / "policy.pt"
        try:
            self.model = torch.jit.load(str(path), map_location="cpu").eval()
        except Exception as error:
            raise RuntimeError(f"Cannot load TorchScript policy {path}; export policy.pt, not a training checkpoint") from error
        self.history = None
        self.previous_actions = np.zeros(12, dtype=np.float32)
        # Fail at startup if the exported policy does not match this task.
        self.infer(np.zeros(225, dtype=np.float32))

    def infer(self, observation):
        torch = self.torch
        observation = finite(observation, (225,))
        with torch.inference_mode():
            output = self.model(torch.from_numpy(observation).reshape(1, 225))
        if not isinstance(output, torch.Tensor) or output.dtype != torch.float32 or tuple(output.shape) != (1, 12):
            raise ValueError("Expected float32 policy output [1,12] for input [1,225]")
        return finite(output.detach().cpu().numpy()[0], (12,)).copy()

    def observation(self, angular_velocity, orientation, names, positions, velocities, command, reset=False):
        angular_velocity = finite(angular_velocity, (3,))
        quaternion = finite(orientation, (4,)).astype(np.float64)
        norm = np.linalg.norm(quaternion)
        if norm < 1e-6:
            raise ValueError("Orientation quaternion has zero norm")
        x, y, z, w = quaternion / norm
        positions = finite(positions, (12,))
        velocities = finite(velocities, (12,))
        expected = [joint["name"] for joint in self.config["joints"]]
        if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected):
            raise ValueError("State must contain all twelve unique configured joint names")
        order = [names.index(name) for name in expected]
        frame = np.concatenate((
            0.2 * angular_velocity,
            [2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1],
            finite(command, (3,)), positions[order] - self.defaults,
            0.05 * velocities[order], np.zeros(12) if reset else self.previous_actions,
        )).astype(np.float32)
        finite(frame, (45,))
        history = (np.tile(frame, (5, 1)) if reset or self.history is None
                   else np.concatenate((self.history[1:], frame[None, :]), axis=0))
        # Isaac Lab flattens each term's history, oldest first, then joins terms.
        observation = np.concatenate([history[:, start:end].ravel() for start, end in
                                      ((0, 3), (3, 6), (6, 9), (9, 21), (21, 33), (33, 45))])
        return observation, history

    def step(self, *args, reset=False):
        observation, history = self.observation(*args, reset=reset)
        actions = self.infer(observation)
        targets = finite(self.defaults + self.config["action_scale"] * actions, (12,))
        # Failed validation or inference must not advance history.
        self.history = history
        self.previous_actions = actions.copy()
        return actions, targets


class JoyControls:
    def __init__(self, config):
        self.config = config
        self.clear()

    def clear(self):
        self.started = False
        self.enabled = False
        self.reset_history = True
        self.buttons = None
        self.command = np.zeros(3, dtype=np.float32)

    def update(self, axes, buttons):
        config = self.config
        indices = config["buttons"]
        if len(axes) <= max(config["axes"]) or len(buttons) <= max(indices.values()):
            raise ValueError("Joy message is too short for configured axis/button mappings")
        values = finite([axes[i] for i in config["axes"]], (3,))
        values = np.clip(values, -1, 1)
        magnitude = np.maximum(0, np.abs(values) - config["deadzone"]) / (1 - config["deadzone"])
        self.command = np.sign(values) * magnitude * finite(config["scales"], (3,))
        current = {key: bool(buttons[index]) for key, index in indices.items()}
        previous = self.buttons
        self.buttons = current
        # A controller already held at connection must not arm the policy.
        if previous is None:
            return
        combo = current["l1"] and current["r1"]
        previous_combo = previous["l1"] and previous["r1"]
        if current["stop"]:
            self.started = self.enabled = False
            self.reset_history = True
        else:
            if current["start"] and not previous["start"] and not self.started:
                self.started = True
                self.reset_history = True
            if self.started and combo and not previous_combo:
                self.enabled = True
