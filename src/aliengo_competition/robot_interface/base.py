from __future__ import annotations

from abc import ABC, abstractmethod


class AliengoRobotInterface(ABC):
    @abstractmethod
    def set_speed(self, vx: float, vy: float, vw: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_body_pitch(self, pitch: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def get_camera(self):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
        raise NotImplementedError

    @abstractmethod
    def is_fallen(self) -> bool:
        raise NotImplementedError
