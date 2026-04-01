from __future__ import annotations

from types import SimpleNamespace

from aliengo_competition.common.helpers import get_args
from aliengo_competition.common.task_registry import task_registry
from aliengo_competition import envs as _registered_envs  # noqa: F401
from aliengo_competition.robot_interface.sim import SimAliengoRobot


def _clone_args(args, *, task: str, mode: str, headless: bool, load_run=-1, checkpoint=-1):
    clone = SimpleNamespace(**vars(args))
    clone.task = task
    clone.mode = mode
    clone.headless = headless
    clone.num_envs = 1
    clone.load_run = load_run
    clone.checkpoint = checkpoint
    clone.resume = True
    return clone


def make_robot_interface(
    *,
    args=None,
    task: str = "aliengo_flat",
    mode: str = "sim",
    headless: bool = True,
    load_run=-1,
    checkpoint=-1,
):
    if args is None:
        args = get_args()
    cloned_args = _clone_args(args, task=task, mode=mode, headless=headless, load_run=load_run, checkpoint=checkpoint)

    env_cfg, train_cfg = task_registry.get_cfgs(task)
    env_cfg.env.num_envs = 1
    env_cfg.seed = train_cfg.seed
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = cloned_args.load_run
    train_cfg.runner.checkpoint = cloned_args.checkpoint

    env, _ = task_registry.make_env(task, args=cloned_args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env, name=task, args=cloned_args, train_cfg=train_cfg, log_root=None)
    policy = runner.get_inference_policy(device=env.device)

    return SimAliengoRobot(env=env, policy=policy)
