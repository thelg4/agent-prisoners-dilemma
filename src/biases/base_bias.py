from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseBias(ABC):
    """Abstract base class for cognitive biases"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bias_name = self.__class__.__name__.lower()
    
    @abstractmethod
    def apply_bias(self, value: float, context: Dict[str, Any]) -> float:
        """Apply the bias to a given value"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of this bias"""
        pass