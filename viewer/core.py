import abc
import torch

class BaseSimulationData(abc.ABC):
    """Абстрактний клас для даних симуляції"""
    @abc.abstractmethod
    def find_dynamic_edges(self, radius_mult: float):
        pass

    @abc.abstractmethod
    def step_physics(self, accel: torch.Tensor):
        pass

class BasePredictor(abc.ABC):
    """Абстрактний клас для моделі передбачення"""
    @abc.abstractmethod
    def predict(self, sim_data: BaseSimulationData, radius_mult: float) -> torch.Tensor:
        pass

class BaseVisualizer(abc.ABC):
    """Абстрактний клас для візуалізатора"""
    @abc.abstractmethod
    def load_scenario(self, idx: int):
        pass
        
    @abc.abstractmethod
    def generate_step(self):
        pass
        
    @abc.abstractmethod
    def run(self):
        pass