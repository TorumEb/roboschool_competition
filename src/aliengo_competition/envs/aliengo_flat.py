from __future__ import annotations

from collections.abc import Sequence
import math
import os

import numpy as np
from isaacgym import gymapi, gymtorch
import torch

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from isaacgym.torch_utils import torch_rand_float, to_torch

from aliengo_competition.common.base_task import BaseTask
from aliengo_competition.common.command_curriculum import RewardThresholdCurriculum
from aliengo_competition.common.math import projected_gravity_xyzw, quat_rotate_inverse_xyzw


class AliengoFlatEnv(BaseTask):
    CMD_VX = 0
    CMD_VY = 1
    CMD_VW = 2
    CMD_BODY_HEIGHT = 3
    CMD_GAIT_FREQUENCY = 4
    CMD_GAIT_PHASE = 5
    CMD_GAIT_OFFSET = 6
    CMD_GAIT_BOUND = 7
    CMD_GAIT_DURATION = 8
    CMD_FOOTSWING_HEIGHT = 9
    CMD_BODY_PITCH = 10
    CMD_BODY_ROLL = 11
    CMD_STANCE_WIDTH = 12
    CMD_STANCE_LENGTH = 13
    CMD_AUX_REWARD = 14

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.manual_command = None
        self._manual_command_tensor = None
        self.default_command = None
        self.command_curriculum = None
        self.command_bins = None
        self._parse_cfg(self.cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._init_command_distribution()
        self._prepare_reward_function()
        self.init_done = True

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_command(self, vx: float, vy: float, vw: float, pitch: float) -> None:
        command = self.default_command.clone()
        command[self.CMD_VX] = float(vx)
        command[self.CMD_VY] = float(vy)
        command[self.CMD_VW] = float(vw)
        command[self.CMD_BODY_PITCH] = float(pitch)
        command[self.CMD_BODY_ROLL] = float(self.default_command[self.CMD_BODY_ROLL].item())
        self.manual_command = command
        self._manual_command_tensor = command.unsqueeze(0).repeat(self.num_envs, 1)
        self.commands[:] = self._manual_command_tensor

    def clear_command_override(self) -> None:
        self.manual_command = None
        self._manual_command_tensor = None

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation.")
        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _joint_gain(self, name: str, gains: dict[str, float]) -> float:
        if name in gains:
            return gains[name]
        if "joint" in gains:
            return gains["joint"]
        return next(iter(gains.values()))

    def _create_envs(self):
        asset_path = self.cfg.asset.file
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)

        self.feet_names = [name for name in body_names if self.cfg.asset.foot_name in name]
        self.penalized_contact_names = []
        for pattern in self.cfg.asset.penalize_contacts_on:
            self.penalized_contact_names.extend([name for name in body_names if pattern in name])
        self.termination_contact_names = []
        for pattern in self.cfg.asset.terminate_after_contacts_on:
            self.termination_contact_names.extend([name for name in body_names if pattern in name])

        self.default_dof_pos = torch.zeros(self.num_dof, device=self.device)
        for i, dof_name in enumerate(self.dof_names):
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles.get(dof_name, 0.0)

        self.p_gains = torch.zeros(self.num_dof, device=self.device)
        self.d_gains = torch.zeros(self.num_dof, device=self.device)
        for i, dof_name in enumerate(self.dof_names):
            self.p_gains[i] = self._joint_gain(dof_name, self.cfg.control.stiffness)
            self.d_gains[i] = self._joint_gain(dof_name, self.cfg.control.damping)

        dof_props = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dof, device=self.device)
        self.torque_limits = torch.zeros(self.num_dof, device=self.device)
        for i in range(len(dof_props)):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            self.torque_limits[i] = dof_props["effort"][i].item()
            midpoint = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2.0
            radius = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = midpoint - 0.5 * radius * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = midpoint + 0.5 * radius * self.cfg.rewards.soft_dof_pos_limit

        self.base_init_state = to_torch(
            self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel,
            device=self.device,
        )
        self.envs = []
        self.actor_handles = []
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)

        num_cols = int(math.ceil(math.sqrt(self.num_envs)))
        num_rows = int(math.ceil(self.num_envs / num_cols))
        lower = gymapi.Vec3(0.0, 0.0, 0.0)
        upper = gymapi.Vec3(0.0, 0.0, 0.0)
        spacing = self.cfg.env.env_spacing

        for env_id in range(self.num_envs):
            row = env_id // num_cols
            col = env_id % num_cols
            self.env_origins[env_id, 0] = row * spacing
            self.env_origins[env_id, 1] = col * spacing
            self.env_origins[env_id, 2] = 0.0

            env_handle = self.gym.create_env(self.sim, lower, upper, num_cols)
            start_pose = gymapi.Transform()
            start_pose.p = gymapi.Vec3(
                float(self.env_origins[env_id, 0].item() + self.base_init_state[0].item()),
                float(self.env_origins[env_id, 1].item() + self.base_init_state[1].item()),
                float(self.base_init_state[2].item()),
            )
            start_pose.r = gymapi.Quat(self.base_init_state[3].item(), self.base_init_state[4].item(), self.base_init_state[5].item(), self.base_init_state[6].item())

            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name, env_id, self.cfg.asset.self_collisions, 0)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.feet_names):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
        self.penalised_contact_indices = torch.zeros(len(self.penalized_contact_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.penalized_contact_names):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
        self.termination_contact_indices = torch.zeros(len(self.termination_contact_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.termination_contact_names):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = {
            key: getattr(self.cfg.rewards.scales, key)
            for key in dir(self.cfg.rewards.scales)
            if not key.startswith("_") and not callable(getattr(self.cfg.rewards.scales, key))
        }
        self.command_ranges = {
            "lin_vel_x": self.cfg.commands.lin_vel_x,
            "lin_vel_y": self.cfg.commands.lin_vel_y,
            "ang_vel_yaw": self.cfg.commands.ang_vel_yaw,
            "body_height": self.cfg.commands.body_height,
            "gait_frequency": self.cfg.commands.gait_frequency,
            "gait_phase": self.cfg.commands.gait_phase,
            "gait_offset": self.cfg.commands.gait_offset,
            "gait_bound": self.cfg.commands.gait_bound,
            "gait_duration": self.cfg.commands.gait_duration,
            "footswing_height": self.cfg.commands.footswing_height,
            "body_pitch": self.cfg.commands.body_pitch,
            "body_roll": self.cfg.commands.body_roll,
            "stance_width": self.cfg.commands.stance_width,
            "stance_length": self.cfg.commands.stance_length,
            "aux_reward_coef": self.cfg.commands.aux_reward_coef,
        }
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(math.ceil(self.max_episode_length_s / self.dt))

    def _init_buffers(self):
        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, -1, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, 0:3]
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self.default_command = to_torch(self.cfg.commands.default_command, device=self.device)
        self.commands = self.default_command.unsqueeze(0).repeat(self.num_envs, 1).clone()
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.body_height_cmd,
                self.obs_scales.gait_freq_cmd,
                self.obs_scales.gait_phase_cmd,
                self.obs_scales.gait_offset_cmd,
                self.obs_scales.gait_bound_cmd,
                self.obs_scales.gait_duration_cmd,
                self.obs_scales.footswing_height_cmd,
                self.obs_scales.body_pitch_cmd,
                self.obs_scales.body_roll_cmd,
                self.obs_scales.stance_width_cmd,
                self.obs_scales.stance_length_cmd,
                self.obs_scales.aux_reward_cmd,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self.gait_indices = torch.zeros(self.num_envs, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)
        self.foot_indices = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, device=self.device)
        self.desired_footswing_height = torch.full((self.num_envs,), float(self.default_command[self.CMD_FOOTSWING_HEIGHT].item()), device=self.device)
        self.foot_positions = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        self.prev_foot_positions = torch.zeros_like(self.foot_positions)
        self.foot_velocities = torch.zeros_like(self.foot_positions)
        self.prev_foot_velocities = torch.zeros_like(self.foot_positions)
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device, dtype=torch.bool)
        self.command_bins = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.common_step_counter = 0
        self.gravity_vec = to_torch([0.0, 0.0, -1.0], device=self.device).repeat((self.num_envs, 1))
        self.zero_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)

    def _init_command_distribution(self):
        if not getattr(self.cfg.commands, "command_curriculum", False):
            return
        self.command_curriculum = RewardThresholdCurriculum(
            seed=self.cfg.commands.curriculum_seed,
            x_vel=(self.cfg.commands.limit_vel_x[0], self.cfg.commands.limit_vel_x[1], self.cfg.commands.num_bins_vel_x),
            y_vel=(self.cfg.commands.limit_vel_y[0], self.cfg.commands.limit_vel_y[1], self.cfg.commands.num_bins_vel_y),
            yaw_vel=(self.cfg.commands.limit_vel_yaw[0], self.cfg.commands.limit_vel_yaw[1], self.cfg.commands.num_bins_vel_yaw),
            body_pitch=(self.cfg.commands.limit_body_pitch[0], self.cfg.commands.limit_body_pitch[1], self.cfg.commands.num_bins_body_pitch),
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(device=self.device)
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.0
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        yaw = torch_rand_float(-math.pi, math.pi, (len(env_ids), 1), device=self.device)
        half = 0.5 * yaw.squeeze(1)
        self.root_states[env_ids, 3] = 0.0
        self.root_states[env_ids, 4] = 0.0
        self.root_states[env_ids, 5] = torch.sin(half)
        self.root_states[env_ids, 6] = torch.cos(half)
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self._update_command_curriculum(env_ids)
        if self.manual_command is not None and self._manual_command_tensor is not None:
            self.commands[env_ids] = self._manual_command_tensor[env_ids]
        else:
            self._resample_commands(env_ids)
        self.gait_indices[env_ids] = 0.0
        self.foot_indices[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0
        self.desired_footswing_height[env_ids] = self.commands[env_ids, self.CMD_FOOTSWING_HEIGHT]

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][f"rew_{key}"] = torch.mean(self.episode_sums[key][env_ids]) / max(self.max_episode_length_s, 1.0)
            self.episode_sums[key][env_ids] = 0.0

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

    def _update_command_curriculum(self, env_ids):
        if self.command_curriculum is None or len(env_ids) == 0 or self.manual_command is not None:
            return
        if torch.any(self.command_bins[env_ids] < 0):
            return
        thresholds = [
            self.cfg.curriculum_thresholds.tracking_lin_vel,
            self.cfg.curriculum_thresholds.tracking_ang_vel,
            self.cfg.curriculum_thresholds.tracking_body_pitch,
        ]
        rewards = [
            self.episode_sums["tracking_lin_vel"][env_ids] / max(self.max_episode_length_s, 1.0),
            self.episode_sums["tracking_ang_vel"][env_ids] / max(self.max_episode_length_s, 1.0),
            self.episode_sums["tracking_body_pitch"][env_ids] / max(self.max_episode_length_s, 1.0),
        ]
        self.command_curriculum.update(
            self.command_bins[env_ids].detach().cpu().numpy(),
            [reward.detach().cpu().numpy() for reward in rewards],
            thresholds,
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.manual_command is not None and self._manual_command_tensor is not None:
            self.commands[env_ids] = self._manual_command_tensor[env_ids]
            return
        self.commands[env_ids] = self.default_command.unsqueeze(0).repeat(len(env_ids), 1)
        if self.command_curriculum is None:
            self.commands[env_ids, self.CMD_VX] = torch_rand_float(*self.command_ranges["lin_vel_x"], (len(env_ids),), device=self.device)
            self.commands[env_ids, self.CMD_VY] = torch_rand_float(*self.command_ranges["lin_vel_y"], (len(env_ids),), device=self.device)
            self.commands[env_ids, self.CMD_VW] = torch_rand_float(*self.command_ranges["ang_vel_yaw"], (len(env_ids),), device=self.device)
            self.commands[env_ids, self.CMD_BODY_PITCH] = torch_rand_float(*self.command_ranges["body_pitch"], (len(env_ids),), device=self.device)
            self.command_bins[env_ids] = -1
            return

        sampled, bins = self.command_curriculum.sample(len(env_ids))
        sampled = torch.as_tensor(sampled, device=self.device, dtype=torch.float32)
        self.commands[env_ids, self.CMD_VX] = sampled[:, 0]
        self.commands[env_ids, self.CMD_VY] = sampled[:, 1]
        self.commands[env_ids, self.CMD_VW] = sampled[:, 2]
        self.commands[env_ids, self.CMD_BODY_PITCH] = sampled[:, 3]
        self.command_bins[env_ids] = torch.as_tensor(bins, device=self.device, dtype=torch.long)

    def _compute_torques(self, actions):
        action_scale = self.cfg.control.action_scale
        target = actions * action_scale + self.default_dof_pos - self.dof_pos
        if self.cfg.control.control_type == "P":
            torques = self.p_gains * target - self.d_gains * self.dof_vel
        elif self.cfg.control.control_type == "T":
            torques = actions * action_scale
        else:
            raise ValueError(f"Unsupported control type: {self.cfg.control.control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if not self.cfg.env.observe_gait_commands:
            return

        frequencies = self.commands[:, self.CMD_GAIT_FREQUENCY]
        phases = self.commands[:, self.CMD_GAIT_PHASE]
        offsets = self.commands[:, self.CMD_GAIT_OFFSET]
        bounds = self.commands[:, self.CMD_GAIT_BOUND]
        durations = torch.clamp(self.commands[:, self.CMD_GAIT_DURATION], 0.05, 0.95)
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if getattr(self.cfg.commands, "pacing_offset", False):
            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + bounds,
                self.gait_indices + offsets,
                self.gait_indices + phases,
            ]
        else:
            foot_indices = [
                self.gait_indices + phases + offsets + bounds,
                self.gait_indices + offsets,
                self.gait_indices + bounds,
                self.gait_indices + phases,
            ]

        self.foot_indices = torch.remainder(torch.cat([idx.unsqueeze(1) for idx in foot_indices], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1.0) < durations
            swing_idxs = torch.remainder(idxs, 1.0) >= durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1.0) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1.0) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs])
            )

        self.clock_inputs[:, 0] = torch.sin(2 * math.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * math.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * math.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * math.pi * foot_indices[3])
        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * math.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * math.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * math.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * math.pi * foot_indices[3])
        self.halftime_clock_inputs[:, 0] = torch.sin(math.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(math.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(math.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(math.pi * foot_indices[3])

        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        def smooth_contact(foot_index: torch.Tensor) -> torch.Tensor:
            phase = torch.remainder(foot_index, 1.0)
            return (
                smoothing_cdf_start(phase) * (1 - smoothing_cdf_start(phase - 0.5))
                + smoothing_cdf_start(phase - 1.0) * (1 - smoothing_cdf_start(phase - 1.5))
            )

        self.desired_contact_states[:, 0] = smooth_contact(foot_indices[0])
        self.desired_contact_states[:, 1] = smooth_contact(foot_indices[1])
        self.desired_contact_states[:, 2] = smooth_contact(foot_indices[2])
        self.desired_contact_states[:, 3] = smooth_contact(foot_indices[3])
        self.desired_footswing_height = self.commands[:, self.CMD_FOOTSWING_HEIGHT]

    def check_termination(self):
        if len(self.termination_contact_indices) > 0:
            self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        else:
            self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i, name in enumerate(self.reward_names):
            reward = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += reward
            self.episode_sums[name] += reward
        self.episode_sums["total"] += self.rew_buf

    def compute_observations(self):
        self.obs_buf = torch.cat(
            (
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf, self.last_actions), dim=-1)

        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)

        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf, self.clock_inputs), dim=-1)

        if self.cfg.env.observe_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.base_ang_vel * self.obs_scales.ang_vel, self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)

    def _prepare_reward_function(self):
        self.reward_names = []
        self.reward_functions = []
        self.episode_sums = {}
        for name, scale in self.reward_scales.items():
            if scale == 0.0:
                continue
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, f"_reward_{name}"))
            self.episode_sums[name] = torch.zeros(self.num_envs, device=self.device)
        self.episode_sums["total"] = torch.zeros(self.num_envs, device=self.device)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse_xyzw(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse_xyzw(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = projected_gravity_xyzw(self.base_quat)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.prev_foot_positions[:] = self.foot_positions[:]
        self.prev_foot_velocities[:] = self.foot_velocities[:]
        self.foot_positions[:] = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.foot_velocities[:] = self.rigid_body_states[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # Rewards.
    def _reward_tracking_lin_vel(self):
        error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_body_pitch(self):
        current_pitch = torch.atan2(-self.projected_gravity[:, 0], -self.projected_gravity[:, 2])
        current_roll = torch.atan2(self.projected_gravity[:, 1], -self.projected_gravity[:, 2])
        target_pitch = self.commands[:, self.CMD_BODY_PITCH]
        target_roll = self.commands[:, self.CMD_BODY_ROLL]
        error = torch.square(current_pitch - target_pitch) + torch.square(current_roll - target_roll)
        return torch.exp(-error / self.cfg.rewards.orientation_sigma)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        target = self.cfg.rewards.base_height_target + self.commands[:, self.CMD_BODY_HEIGHT]
        return torch.square(self.root_states[:, 2] - target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        if len(self.penalised_contact_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).float(), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        self.last_contacts = contact
        return rew

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0.0
        for i in range(4):
            reward += -(1 - desired_contact[:, i]) * (
                1 - torch.exp(-1.0 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
            )
        return reward / 4.0

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=2)
        desired_contact = self.desired_contact_states
        reward = 0.0
        for i in range(4):
            reward += -(desired_contact[:, i] * (
                1 - torch.exp(-1.0 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)
            ))
        return reward / 4.0

    def _reward_feet_slip(self):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        foot_vel_xy = torch.sum(torch.square(self.foot_velocities[:, :, :2]), dim=2)
        slip = torch.sum(foot_vel_xy * contact.float(), dim=1)
        return slip

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.foot_positions[:, :, 2]
        target_height = self.desired_footswing_height.unsqueeze(1) * phases + 0.02
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        reward = torch.sum(rew_foot_clearance, dim=1)

        def _smoothstep(x):
            return x * x * (3.0 - 2.0 * x)

        v = torch.norm(self.commands[:, :2], dim=1)
        v_on = self.cfg.rewards.stand_vel_on
        v_full = self.cfg.rewards.stand_vel_full
        x = torch.clamp((v - v_on) / (v_full - v_on), 0.0, 1.0)
        reward = reward * _smoothstep(x)
        return reward

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = quat_rotate_inverse_xyzw(
            self.base_quat.unsqueeze(1).expand(-1, len(self.feet_indices), -1).reshape(-1, 4),
            cur_footsteps_translated.reshape(-1, 3),
        ).reshape(self.num_envs, len(self.feet_indices), 3)

        desired_stance_width = self.commands[:, self.CMD_STANCE_WIDTH].unsqueeze(1)
        desired_ys_nom = torch.cat(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            dim=1,
        )
        desired_stance_length = self.commands[:, self.CMD_STANCE_LENGTH].unsqueeze(1)
        desired_xs_nom = torch.cat(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            dim=1,
        )

        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) - 0.5
        frequencies = torch.clamp(self.commands[:, self.CMD_GAIT_FREQUENCY], min=0.1)
        x_vel_des = self.commands[:, self.CMD_VX].unsqueeze(1)
        yaw_vel_des = self.commands[:, self.CMD_VW].unsqueeze(1)
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset
        desired_footsteps_body_frame = torch.stack((desired_xs_nom, desired_ys_nom), dim=2)
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        def _smoothstep(x):
            return x * x * (3.0 - 2.0 * x)

        v = torch.norm(self.commands[:, :2], dim=1)
        v_on = self.cfg.rewards.stand_vel_on
        v_full = self.cfg.rewards.stand_vel_full
        x = torch.clamp((v - v_on) / (v_full - v_on), 0.0, 1.0)
        reward = reward * _smoothstep(x)
        return reward

    def _reward_stand_still(self):
        def _smoothstep(x):
            return x * x * (3.0 - 2.0 * x)

        v = torch.norm(self.commands[:, :2], dim=1)
        v_off = self.cfg.rewards.stand_vel_on
        v_full = self.cfg.rewards.stand_vel_full
        x = torch.clamp((v - v_off) / (v_full - v_off), 0.0, 1.0)
        w_walk = _smoothstep(x)
        w_stand = 1.0 - w_walk

        pos_cost = torch.sum(torch.abs(self.actions), dim=1)
        vel_cost = torch.sum(torch.square(self.dof_vel), dim=1)
        return w_stand * (0.2 * pos_cost + 0.5 * vel_cost)
