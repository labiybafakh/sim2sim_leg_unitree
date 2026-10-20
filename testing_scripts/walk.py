import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

# All joint arrays below use this policy/training order, not MuJoCo XML order.
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",     # 0
    "right_hip_pitch_joint",    # 1
    "left_hip_roll_joint",      # 2
    "right_hip_roll_joint",     # 3
    "left_hip_yaw_joint",       # 4
    "right_hip_yaw_joint",      # 5
    "left_knee_joint",          # 6
    "right_knee_joint",         # 7
    "left_ankle_pitch_joint",   # 8
    "right_ankle_pitch_joint",  # 9
    "left_ankle_roll_joint",    # 10
    "right_ankle_roll_joint",   # 11
]

# MuJoCo motor order: left leg first, then right leg.
SIM_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# Values from the training configuration: radians, PD gains, and Nm limits.
DEFAULT_POSITION = np.array([-0.1, -0.1, 0, 0, 0, 0, 0.3, 0.3, -0.2, -0.2, 0, 0], dtype=np.float32)
KP = np.array([100, 100, 100, 100, 100, 100, 150, 150, 40, 40, 40, 40])
KD = np.array([2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2])
TORQUE_LIMIT = np.array([88, 88, 139, 139, 88, 88, 139, 139, 25, 25, 25, 25])
TIMESTEP = 0.005  # 200 Hz physics
DECIMATION = 4    # 50 Hz policy
ACTION_SCALE = 0.25
COMMAND = np.array([0.3, 0.0, 0.0], dtype=np.float32)  # vx, vy, omega
DURATION = 20.0
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "policy_deployment/models/policy.pt"


# Slice stops are excluded. Sizes count scalar values, not the batch dimension.
POLICY_INPUT = {
    "shape": (1, 225),
    "dtype": "float32",
    "size": 225,
    "frame_size": 45,
    "history_length": 5,
    "terms": {
        "angular_velocity": {"frame_slice": slice(0, 3), "slice": slice(0, 15), "frame_size": 3, "size": 15},
        "projected_gravity": {"frame_slice": slice(3, 6), "slice": slice(15, 30), "frame_size": 3, "size": 15},
        "commands": {"frame_slice": slice(6, 9), "slice": slice(30, 45), "frame_size": 3, "size": 15},
        "joint_positions": {"frame_slice": slice(9, 21), "slice": slice(45, 105), "frame_size": 12, "size": 60},
        "joint_velocities": {"frame_slice": slice(21, 33), "slice": slice(105, 165), "frame_size": 12, "size": 60},
        "previous_actions": {"frame_slice": slice(33, 45), "slice": slice(165, 225), "frame_size": 12, "size": 60},
    },
}
POLICY_OUTPUT = {
    "shape": (1, 12),
    "dtype": "float32",
    "size": 12,
    "slice": slice(0, 12),
    "joint_indices": {name: index for index, name in enumerate(POLICY_JOINT_NAMES)},
}


def get_joint_index_map(policy_to_sim, qpos_indices, qvel_indices):
    """Return named indices, including every history slot for each joint."""
    joints = {}
    for name, index in POLICY_OUTPUT["joint_indices"].items():
        joints[name] = {
            "action": index,
            "motor": int(policy_to_sim[index]),
            "qpos": int(qpos_indices[index]),
            "qvel": int(qvel_indices[index]),
            "observations": {
                term: [POLICY_INPUT["terms"][term]["slice"].start + 12*h + index
                       for h in range(POLICY_INPUT["history_length"])]
                for term in ("joint_positions", "joint_velocities", "previous_actions")
            },
        }
    return joints


def print_index_layout(policy_to_sim, qpos_indices, qvel_indices):
    """Report zero-based indices before filling state, observations or controls."""
    print(f"\nPolicy input: shape={POLICY_INPUT['shape']}, size={POLICY_INPUT['size']}, "
          f"dtype={POLICY_INPUT['dtype']}")
    print(f"Policy output: shape={POLICY_OUTPUT['shape']}, size={POLICY_OUTPUT['size']}, "
          f"dtype={POLICY_OUTPUT['dtype']}")
    print("Observation slices are zero-based, end excluded; history is oldest first.")
    print(f"{'Term':<20} {'Frame':>8} {'Input slice':>12} {'Size':>5} {'Newest sample':>14}")
    for name, term in POLICY_INPUT["terms"].items():
        frame, indices = term["frame_slice"], term["slice"]
        newest = indices.stop - term["frame_size"]
        print(f"{name:<20} {f'{frame.start}:{frame.stop}':>8} "
              f"{f'{indices.start}:{indices.stop}':>12} {term['size']:5d} "
              f"{f'{newest}:{indices.stop}':>14}")
    print("\nJoint indices (observation columns show the NEWEST history sample):")
    print(f"{'Joint':<25} {'Action':>6} {'Motor':>5} {'qpos':>4} {'qvel':>4} "
          f"{'Obs pos':>7} {'Obs vel':>7} {'Obs prev':>8}")
    joint_indices = get_joint_index_map(policy_to_sim, qpos_indices, qvel_indices)
    for name, joint in joint_indices.items():
        obs = joint["observations"]
        print(f"{name:<25} {joint['action']:6d} {joint['motor']:5d} "
              f"{joint['qpos']:4d} {joint['qvel']:4d} "
              f"{obs['joint_positions'][-1]:7d} {obs['joint_velocities'][-1]:7d} "
              f"{obs['previous_actions'][-1]:8d}")
    print()


def setup_simulation():
    """Load the scene, map joint indices, and initialize the standing pose."""
    # Load the robot and map each policy joint to its actual MuJoCo indices.
    model = mujoco.MjModel.from_xml_path(str(ROOT / "unitree_leg_mujoco/scene_29dof.xml"))
    model.opt.timestep = TIMESTEP
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    pelvis = model.body("pelvis").id
    actual_sim_order = [model.joint(int(joint)).name for joint in model.actuator_trnid[:, 0]]
    if actual_sim_order != SIM_JOINT_NAMES:
        raise ValueError("SIM_JOINT_NAMES must match the MuJoCo motor order")
    # For each policy slot, find the matching simulation slot.
    # [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11] for this model.
    policy_to_sim = np.array([SIM_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES])
    sim_joint_ids = np.array([model.joint(name).id for name in SIM_JOINT_NAMES])
    joint_ids = sim_joint_ids[policy_to_sim]
    # qpos/qvel include the floating base: use addresses, not joint IDs as indices.
    qpos_indices = model.jnt_qposadr[joint_ids]
    qvel_indices = model.jnt_dofadr[joint_ids]

    print_index_layout(policy_to_sim, qpos_indices, qvel_indices)

    # Override stock XML effort limits to match training.
    limits = np.column_stack((-TORQUE_LIMIT, TORQUE_LIMIT))
    model.actuator_ctrlrange[policy_to_sim] = limits
    model.actuator_ctrllimited[policy_to_sim] = True
    model.jnt_actfrcrange[joint_ids] = limits
    model.jnt_actfrclimited[joint_ids] = True
    data.qpos[qpos_indices] = DEFAULT_POSITION
    mujoco.mj_forward(model, data)

    return model, data, pelvis, policy_to_sim, qpos_indices, qvel_indices


def build_observation(model, data, qpos_indices, qvel_indices, previous_action, history):
    """Read the robot state and update five samples of observation history."""
    pelvis = model.body("pelvis").id
    # Local link-frame angular velocity; do not use the inertia frame.
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                            pelvis, velocity, 1)
    # Inverse-rotate world down into the pelvis frame: R.T @ [0,0,-1].
    rotation = data.xmat[pelvis].reshape(3, 3)
    projected_gravity = -rotation[2, :]
    frame = np.concatenate((
        0.2 * velocity[:3],
        projected_gravity,
        COMMAND,
        data.qpos[qpos_indices] - DEFAULT_POSITION,  # simulation -> policy
        0.05 * data.qvel[qvel_indices],
        previous_action,
    )).astype(np.float32)
    if history is None:
        history = np.tile(frame, (POLICY_INPUT["history_length"], 1))
    else:
        history[:-1] = history[1:].copy()
        history[-1] = frame

    observation = np.concatenate([
        history[:, term["frame_slice"]].ravel()
        for term in POLICY_INPUT["terms"].values()
    ])
    return observation, history


def run_policy(policy, observation):
    with torch.inference_mode():
        output = policy(torch.from_numpy(observation).reshape(POLICY_INPUT["shape"]))
    return output.numpy()[0].copy()


def apply_pd_control(data, qpos_indices, qvel_indices, policy_to_sim, target_position):
    """Compute torques in policy order, then write them in simulation order."""
    torque = KP * (target_position - data.qpos[qpos_indices]) - KD * data.qvel[qvel_indices]
    data.ctrl[policy_to_sim] = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    model, data, pelvis, policy_to_sim, qpos_indices, qvel_indices = setup_simulation()
    torch.set_num_threads(1)
    policy = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    previous_action = np.zeros(POLICY_OUTPUT["size"], dtype=np.float32)
    history = None
    target_position = DEFAULT_POSITION.copy()
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as mujoco_viewer

        context = mujoco_viewer.launch_passive(model, data)

    tick = 0
    with context as viewer:
        if viewer is not None:
            viewer.cam.trackbodyid = pelvis
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -15
        wall_start = time.monotonic()
        while data.time < DURATION:
            if viewer is not None and not viewer.is_running():
                break
            # Infer at 50 Hz; hold the target between policy updates.
            if tick % DECIMATION == 0:
                observation, history = build_observation(
                    model, data, qpos_indices, qvel_indices, previous_action, history)
                action = run_policy(policy, observation)
                target_position = DEFAULT_POSITION + ACTION_SCALE * action
                previous_action = action

            # Apply PD torques and advance physics at 200 Hz.
            apply_pd_control(data, qpos_indices, qvel_indices, policy_to_sim, target_position)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            tick += 1
            if data.xpos[pelvis, 2] < 0.2 or data.xmat[pelvis, 8] < np.cos(0.8):
                raise RuntimeError(f"Robot fell at t={data.time:.3f}s")
            if viewer is not None:
                viewer.sync()
                delay = wall_start + data.time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)


if __name__ == "__main__":
    main()
