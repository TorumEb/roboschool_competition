from .aliengo_flat import AliengoFlatEnv
from configs.aliengo_flat import AliengoFlatCfg, AliengoFlatCfgPPO

from aliengo_competition.common.task_registry import task_registry


task_registry.register("aliengo_flat", AliengoFlatEnv, AliengoFlatCfg(), AliengoFlatCfgPPO())

__all__ = ["AliengoFlatEnv", "AliengoFlatCfg", "AliengoFlatCfgPPO", "task_registry"]

