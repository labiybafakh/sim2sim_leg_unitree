"""Check joint mapping and compare alternate observations with deployment."""

import sys

import mujoco
import numpy as np

from walk import (ROOT, MODEL_PATH, POLICY_JOINT_NAMES, SIM_JOINT_NAMES,
                  DEFAULT_POSITION, ACTION_SCALE, COMMAND, KP, KD, TIMESTEP, DECIMATION,
                  print_index_layout, POLICY_INPUT, POLICY_OUTPUT, get_joint_index_map)
from leg_utils import get_policy_observation, get_policy_output

sys.path.insert(0, str(ROOT / "policy_deployment/scripts"))
from policy_runtime import Policy, load_config


def check_joint_mapping(model):
    """Verify the policy and simulation orders refer to the same joints."""
    actual_order = [model.joint(int(joint)).name for joint in model.actuator_trnid[:, 0]]
    assert actual_order == SIM_JOINT_NAMES
    policy_to_sim = np.array([SIM_JOINT_NAMES.index(name) for name in POLICY_JOINT_NAMES])
    # Check reading simulation values into policy order and writing them back.
    simulation_values = np.arange(12)
    policy_values = simulation_values[policy_to_sim]
    recovered = np.empty(12, dtype=int)
    recovered[policy_to_sim] = policy_values
    np.testing.assert_array_equal(recovered, simulation_values)
    for index, name in enumerate(POLICY_JOINT_NAMES):
        assert actual_order[policy_to_sim[index]] == name

    joint_ids = np.array([model.joint(name).id for name in POLICY_JOINT_NAMES])
    joints = get_joint_index_map(policy_to_sim, model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids])
    for name, joint in joints.items():
        assert actual_order[joint["motor"]] == name
        for term, indices in joint["observations"].items():
            assert len(indices) == POLICY_INPUT["history_length"]
            assert all(POLICY_INPUT["terms"][term]["slice"].start <= i
                       < POLICY_INPUT["terms"][term]["slice"].stop for i in indices)
    return policy_to_sim


def check_configuration(config):
    """Verify walking constants match the deployment configuration."""
    assert [joint["name"] for joint in config["joints"]] == POLICY_JOINT_NAMES
    terms = list(POLICY_INPUT["terms"].values())
    assert sum(term["size"] for term in terms) == POLICY_INPUT["size"]
    assert [i for term in terms for i in range(term["slice"].start, term["slice"].stop)] == list(range(POLICY_INPUT["size"]))
    assert POLICY_OUTPUT["size"] == len(POLICY_JOINT_NAMES)
    np.testing.assert_allclose(DEFAULT_POSITION, [j["default_position"] for j in config["joints"]])
    np.testing.assert_allclose(KP, [j["stiffness"] for j in config["joints"]])
    np.testing.assert_allclose(KD, [j["damping"] for j in config["joints"]])
    np.testing.assert_allclose(ACTION_SCALE, config["action_scale"])
    np.testing.assert_allclose(1 / (TIMESTEP * DECIMATION), config["policy_frequency_hz"])


def print_initial_output(policy):
    """Show inference from an upright default pose, without stepping physics."""
    # A useful reference: what the policy commands from the upright default pose.
    nominal_obs, _ = get_policy_observation(
        [0, 0, 0, 1], np.zeros(3), COMMAND, DEFAULT_POSITION, np.zeros(12),
        DEFAULT_POSITION, np.zeros(12))
    nominal_action = get_policy_output(policy, nominal_obs)
    nominal_target = DEFAULT_POSITION + ACTION_SCALE * nominal_action
    print(f"Input: {nominal_obs.dtype} [1,{nominal_obs.size}]; "
          f"output: {nominal_action.dtype} [1,{nominal_action.size}]")
    print("Upright gravity: [0, 0, -1]; history: 5 samples per term, oldest first.")
    print("\nInitial output from upright default pose (target in radians):")
    print(f"{'Action index':>12} {'Joint':<25} {'Action':>9} {'Target':>9}")
    for index, name in enumerate(POLICY_JOINT_NAMES):
        print(f"{index:12d} {name:<25} {nominal_action[index]:+9.5f} {nominal_target[index]:+9.5f}")



def check_synthetic_states(runtime):
    """Compare history, gravity, reordered joints and actions across eight states."""
    term_names = ("angular velocity", "projected gravity", "commands",
                  "joint positions", "joint velocities", "previous actions")
    term_bounds = (0, 15, 30, 45, 105, 165, 225)
    max_term_error = np.zeros(6)
    max_action_error = max_target_error = 0.0
    history = None
    previous_action = np.zeros(12, dtype=np.float32)
    rng = np.random.default_rng(42)
    for index in range(8):
        reset = index in (0, 5)
        if reset:
            history = None
            previous_action = np.zeros(12, dtype=np.float32)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        angular = rng.normal(size=3)
        positions = DEFAULT_POSITION + rng.uniform(-0.1, 0.1, 12)
        velocities = rng.normal(size=12)
        observation, history = get_policy_observation(
            quat, angular, COMMAND, positions, velocities, DEFAULT_POSITION,
            previous_action, history)
        # Deployment must agree even when incoming joints are reversed.
        state = (angular, quat, POLICY_JOINT_NAMES[::-1], positions[::-1], velocities[::-1], COMMAND)
        expected, _ = runtime.observation(*state, reset=reset)
        np.testing.assert_allclose(observation, expected, rtol=1e-5, atol=1e-6)
        for term, (start, end) in enumerate(zip(term_bounds[:-1], term_bounds[1:])):
            error = np.max(np.abs(observation[start:end] - expected[start:end]))
            max_term_error[term] = max(max_term_error[term], error)
        action = get_policy_output(runtime.model, observation)
        assert observation.shape == (225,) and action.shape == (12,)
        assert np.isfinite(action).all()
        expected_action, targets = runtime.step(*state, reset=reset)
        np.testing.assert_allclose(action, expected_action, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(targets, DEFAULT_POSITION + ACTION_SCALE * action,
                                   rtol=1e-4, atol=1e-5)
        max_action_error = max(max_action_error, float(np.max(np.abs(action - expected_action))))
        target_error = np.max(np.abs(targets - (DEFAULT_POSITION + ACTION_SCALE * action)))
        max_target_error = max(max_target_error, float(target_error))
        previous_action = action
    print("\nMaximum absolute differences from deployment across 8 synthetic states:")
    for term, error in zip(term_names, max_term_error):
        print(f"  {term:<20} {error:.3e}")
    print(f"  {'actions':<20} {max_action_error:.3e}")
    print(f"  {'targets (rad)':<20} {max_target_error:.3e}")
    print("PASS: changing orientation, reversed input joints, history shifts, and 2 history resets.")
    print("This checks policy inputs/outputs; use walk.py to measure physical walking.")


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "unitree_leg_mujoco/scene_29dof.xml"))
    config = load_config(ROOT / "policy_deployment/config/policy.json")
    policy_to_sim = check_joint_mapping(model)
    check_configuration(config)
    runtime = Policy(config, MODEL_PATH)

    print(f"Model: {MODEL_PATH}")
    print(f"Command (vx, vy, omega): {np.array2string(COMMAND, precision=3)}")
    print(f"Physics: {1/TIMESTEP:g} Hz; policy: {1/(TIMESTEP*DECIMATION):g} Hz; action scale: {ACTION_SCALE}")
    print("PASS: default pose, PD gains, action scale, and policy rate match deployment config.")
    joint_ids = np.array([model.joint(name).id for name in POLICY_JOINT_NAMES])
    print_index_layout(policy_to_sim, model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids])
    print_initial_output(runtime.model)
    check_synthetic_states(runtime)


if __name__ == "__main__":
    main()
