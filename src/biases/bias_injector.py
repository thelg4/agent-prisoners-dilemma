from typing import Dict, Any, List, Optional
from .base_bias import BaseBias
from .loss_aversion import LossAversion

class BiasInjector:
    """Manages and applies multiple cognitive biases"""
    
    def __init__(self):
        self.biases: Dict[str, BaseBias] = {}
    
    def add_bias(self, bias_name: str, bias: BaseBias) -> None:
        """Add a bias to the injector"""
        self.biases[bias_name] = bias
    
    def remove_bias(self, bias_name: str) -> None:
        """Remove a bias from the injector"""
        if bias_name in self.biases:
            del self.biases[bias_name]
    
    def apply_all_biases(self, value: float, context: Dict[str, Any]) -> float:
        """Apply all registered biases to a value"""
        result = value
        for bias in self.biases.values():
            result = bias.apply_bias(result, context)
        return result
    
    def get_bias_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all registered biases"""
        return {name: bias.get_description() for name, bias in self.biases.items()}
    
    @classmethod
    def create_loss_averse_injector(cls, loss_coefficient: float = 2.25) -> 'BiasInjector':
        """Create a bias injector with loss aversion"""
        injector = cls()
        loss_aversion = LossAversion({"loss_coefficient": loss_coefficient})
        injector.add_bias("loss_aversion", loss_aversion)
        return injector