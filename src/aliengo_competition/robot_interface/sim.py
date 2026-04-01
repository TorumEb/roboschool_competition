from __future__ import annotations

from dataclasses import dataclass

import torch

from aliengo_competition.robot_interface.base import AliengoRobotInterface


@dataclass
class StepResult:
    observation: torch.Tensor
    reward: torch.Tensor | None
    done: torch.Tensor | None
    info: dict


class SimAliengoRobot(AliengoRobotInterface):
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self._speed = torch.zeros(3, device=self.env.device)
        self._pitch = torch.tensor(0.0, device=self.env.device)
        self._last_result = StepResult(
            observation=self.env.get_observations(),
            reward=None,
            done=None,
            info={},
        )

    def set_speed(self, vx: float, vy: float, vw: float) -> None:
        self._speed = torch.tensor([vx, vy, vw], device=self.env.device, dtype=torch.float32)
        self.env.set_command(float(vx), float(vy), float(vw), float(self._pitch.item()))

    def set_body_pitch(self, pitch: float) -> None:
        self._pitch = torch.tensor(float(pitch), device=self.env.device, dtype=torch.float32)
        self.env.set_command(float(self._speed[0].item()), float(self._speed[1].item()), float(self._speed[2].item()), float(self._pitch.item()))

    def stop(self) -> None:
        self._speed.zero_()
        self._pitch.zero_()
        self.env.set_command(0.0, 0.0, 0.0, 0.0)

    def reset(self):
        obs, privileged_obs = self.env.reset()
        self._last_result = StepResult(observation=obs, reward=None, done=None, info={"privileged_obs": privileged_obs})
        return obs

    def step(self):
        obs = self.env.get_observations()
        action = self.policy(obs.detach())
        obs, privileged_obs, reward, done, info = self.env.step(action.detach())
        self._last_result = StepResult(observation=obs, reward=reward, done=done, info=info)
        self._last_result.info["privileged_obs"] = privileged_obs
        return obs, reward, done, info

    def get_camera(self):
        return None

    def get_observation(self):
        return self._last_result.observation

    def is_fallen(self) -> bool:
        if self._last_result.done is None:
            return False
        return bool(torch.any(self._last_result.done).item())
