from __future__ import annotations

import distutils.version  # noqa: F401
from datetime import datetime
from typing import Tuple

from rsl_rl.runners import OnPolicyRunner

from aliengo_competition import MODELS_DIR
from aliengo_competition.common.helpers import (
    class_to_dict,
    get_args,
    get_load_path,
    parse_sim_params,
    set_seed,
    update_cfg_from_args,
)
from aliengo_competition.common.history_wrapper import HistoryWrapper


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name, task_class, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_cfgs(self, name) -> Tuple[object, object]:
        env_cfg = self.env_cfgs[name]
        train_cfg = self.train_cfgs[name]
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, args=None, env_cfg=None):
        if args is None:
            args = get_args()
        if name not in self.task_classes:
            raise ValueError(f"Task with name '{name}' was not registered")
        task_class = self.task_classes[name]
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        if getattr(env_cfg.env, "num_observation_history", 1) > 1:
            env = HistoryWrapper(env)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        if args is None:
            args = get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be set")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> ignoring name={name}")
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root == "default":
            load_root = MODELS_DIR / train_cfg.runner.experiment_name
            log_dir = load_root / f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{train_cfg.runner.run_name}"
        elif log_root is None:
            load_root = MODELS_DIR / train_cfg.runner.experiment_name
            log_dir = None
        else:
            load_root = log_root
            log_dir = load_root / f"{datetime.now().strftime('%b%d_%H-%M-%S')}_{train_cfg.runner.run_name}"

        runner = OnPolicyRunner(env, self._train_cfg_to_dict(train_cfg), str(log_dir) if log_dir is not None else None, device=args.rl_device)
        runner.log_dir = str(log_dir) if log_dir is not None else str(load_root)
        runner.log_root = str(load_root)
        if train_cfg.runner.resume:
            resume_path = get_load_path(str(load_root), load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

    def _train_cfg_to_dict(self, train_cfg):
        return {
            "seed": train_cfg.seed,
            "runner_class_name": train_cfg.runner_class_name,
            "policy": class_to_dict(train_cfg.policy),
            "algorithm": class_to_dict(train_cfg.algorithm),
            "runner": class_to_dict(train_cfg.runner),
        }


task_registry = TaskRegistry()
