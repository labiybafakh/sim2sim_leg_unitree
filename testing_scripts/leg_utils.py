
import mujoco
import numpy as np
import torch


def get_policy_observation(quat_xyzw, angular_velocity, command, positions,
                           velocities, defaults, previous_action, history=None):
    # Rotate world gravity into the body frame using its orientation quaternion.
    quat = np.asarray(quat_xyzw, dtype=float).copy()
    quat /= np.linalg.norm(quat)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat[[3, 0, 1, 2]])
    gravity = rotation.reshape(3, 3).T @ [0, 0, -1]
    frame = np.concatenate((
        0.2 * angular_velocity,
        gravity,
        command,
        positions - defaults,
        0.05 * velocities,
        previous_action,
    )).astype(np.float32)

    # Joint arrays use policy order; repeat the first frame to initialize history.
    history = np.tile(frame, (5, 1)) if history is None else np.vstack((history[1:], frame))
    observation = np.concatenate([
        history[:, start:end].ravel()
        for start, end in ((0, 3), (3, 6), (6, 9), (9, 21), (21, 33), (33, 45))
    ])
    return observation, history


def get_policy_output(policy, observation):
    with torch.inference_mode():
        return policy(torch.from_numpy(observation).reshape(1, 225)).numpy()[0].copy()
