"""Check reference curriculum thresholds and gait contact rewards without a simulator."""

from types import SimpleNamespace as NS

import torch

from isaaclab.managers import SceneEntityCfg

from unitree_leg.tasks.velocity.mdp.curriculums import lin_vel_cmd_levels
from unitree_leg.tasks.velocity.mdp.rewards import energy, feet_gait


def test_command_curriculum_threshold_and_limits():
    ranges = NS(lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1))
    command = NS(cfg=NS(ranges=ranges, limit_ranges=NS(lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3))))
    sums = torch.tensor([16.0, 16.0])
    env = NS(
        command_manager=NS(get_term=lambda _: command),
        reward_manager=NS(get_term_cfg=lambda _: NS(weight=1.0), _episode_sums={"track_lin_vel_xy": sums}),
        max_episode_length_s=20.0,
        max_episode_length=1000,
        common_step_counter=1000,
        device="cpu",
    )
    lin_vel_cmd_levels(env, [0, 1])
    assert ranges.lin_vel_x == (-0.1, 0.1)
    sums[:] = 18.0
    env.common_step_counter = 999
    lin_vel_cmd_levels(env, [0, 1])
    assert ranges.lin_vel_x == (-0.1, 0.1)
    for step in range(1000, 12000, 1000):
        env.common_step_counter = step
        lin_vel_cmd_levels(env, [0, 1])
    torch.testing.assert_close(torch.tensor(ranges.lin_vel_x), torch.tensor([-0.5, 1.0]))
    torch.testing.assert_close(torch.tensor(ranges.lin_vel_y), torch.tensor([-0.3, 0.3]))


def test_gait_matches_alternating_contacts_and_ignores_standing():
    sensor = NS(data=NS(current_contact_time=NS(torch=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))))
    env = NS(
        scene=NS(sensors={"contact_forces": sensor}),
        episode_length_buf=torch.tensor([0, 21, 0]),
        step_dt=0.02,
        num_envs=3,
        device="cpu",
        command_manager=NS(get_command=lambda _: torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
    )
    cfg = SceneEntityCfg("contact_forces", body_ids=[0, 1])
    result = feet_gait(env, 0.8, [0.0, 0.5], cfg, 0.5, "base_velocity")
    torch.testing.assert_close(result, torch.tensor([2.0, 2.0, 0.0]))


def test_energy_counts_absolute_power():
    robot = NS(
        data=NS(joint_vel=NS(torch=torch.tensor([[2.0, -3.0]]))),
        actuators=NS(applied_effort=NS(torch=torch.tensor([[-4.0, 5.0]]))),
    )
    result = energy(NS(scene={"robot": robot}))
    torch.testing.assert_close(result, torch.tensor([23.0]))
