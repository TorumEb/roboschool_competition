from __future__ import annotations

import time
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import isaacgym

assert isaacgym

import cv2

from aliengo_competition.common.helpers import get_args
from aliengo_competition.robot_interface.factory import make_robot_interface
from sim_bridge_client import SimBridgeClient


DEFAULT_POLICY_RUN = (
    PROJECT_ROOT / "runs" / "gait-conditioned-agility" / "2026-04-01" / "train" / "124256.867761"
)


def main() -> None:
    args = get_args()
    args.headless = False
    if args.load_run in (None, "", -1, "-1"):
        args.load_run = str(DEFAULT_POLICY_RUN)
        print(f"[BridgeSim] Using default policy run: {args.load_run}")

    bridge = SimBridgeClient()
    robot = make_robot_interface(
        args=args,
        task=args.task,
        mode=args.mode,
        headless=args.headless,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    state = robot.reset()

    print("Isaac ROS bridge controller started.")
    print("Receiving cmd on UDP 127.0.0.1:5005")
    print("Sending state on UDP 127.0.0.1:5006")
    print("Sending RGB on UDP 127.0.0.1:5007")
    print("Sending Depth on UDP 127.0.0.1:5008")

    while True:
        loop_start = time.time()
        cmd = bridge.receive_cmd()

        robot.set_velocity_command(
            vx=float(cmd["vx"]),
            vy=float(cmd["vy"]),
            vw=float(cmd["wz"]),
        )
        state = robot.step()

        bridge.send_state(
            vx=float(state.base_linear_velocity_xyz[0]),
            vy=float(state.base_linear_velocity_xyz[1]),
            wz=float(state.base_angular_velocity_xyz[2]),
        )

        if state.camera.rgb is not None:
            rgb_bgr = cv2.cvtColor(state.camera.rgb, cv2.COLOR_RGB2BGR)
            bridge.send_rgb(rgb_bgr)
        if state.camera.depth is not None:
            bridge.send_depth(state.camera.depth)

        if robot.is_fallen():
            robot.stop()
            state = robot.reset()

        dt = robot.get_control_dt()
        elapsed = time.time() - loop_start
        if dt > elapsed:
            time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
