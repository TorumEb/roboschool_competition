from __future__ import annotations

import math

from aliengo_competition.robot_interface.base import AliengoRobotInterface


def run(robot: AliengoRobotInterface, steps: int = 1000) -> None:
    robot.reset()
    phase = 0.0
    for _ in range(steps):
        obs = robot.get_observation()
        if obs.ndim > 1:
            obs = obs[0]
        camera = robot.get_camera()
        # PUT YOUR CODE HERE
        # Minimal example:
        # - read the latest observation
        # - keep a conservative forward walk
        # - soften the command if the robot looks unstable
        # - keep camera access as a stub hook for future vision logic
        _ = camera

        base_obs = obs[-58:] if obs.shape[-1] >= 58 else obs
        projected_gravity = base_obs[0:3]
        pitch_penalty = float(abs(projected_gravity[0]) + abs(projected_gravity[1]))
        vx = 0.35 if pitch_penalty < 0.35 else 0.15
        vy = 0.0
        vw = 0.0
        pitch = 0.04 * math.sin(phase)
        phase += 0.05

        robot.set_speed(vx, vy, vw)
        robot.set_body_pitch(pitch)

        if robot.is_fallen():
            robot.stop()
            robot.reset()
            continue
        robot.step()
