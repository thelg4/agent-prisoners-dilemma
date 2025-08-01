from typing import Dict, Any
from .base_bias import BaseBias
import math

class LossAversion(BaseBias):
    """Implementation of loss aversion bias based on Kahneman & Tversky's prospect theory"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Prospect theory parameters
        self.loss_coefficient = config.get("loss_coefficient", 2.25)  # λ (lambda)
        self.gain_exponent = config.get("gain_exponent", 0.88)        # α (alpha) 
        self.loss_exponent = config.get("loss_exponent", 0.88)        # β (beta)
        
    def apply_bias(self, value: float, reference_point: float) -> float:
        """
        Apply loss aversion bias using prospect theory value function
        
        Args:
            value: The objective value/payoff
            reference_point: The reference point for gains/losses
            
        Returns:
            Subjective value adjusted for loss aversion
        """
        if value >= reference_point:
            # Gains: v(x) = x^α
            gain = value - reference_point
            return reference_point + (gain ** self.gain_exponent)
        else:
            # Losses: v(x) = -λ(-x)^β  
            loss = reference_point - value
            return reference_point - (self.loss_coefficient * (loss ** self.loss_exponent))
    
    def calculate_regret(self, actual_outcome: float, counterfactual_outcome: float) -> float:
        """Calculate regret with loss aversion amplification"""
        if actual_outcome < counterfactual_outcome:
            # We lost out - amplify the regret
            regret = counterfactual_outcome - actual_outcome
            return self.loss_coefficient * regret
        else:
            # We did better than the alternative
            return max(0, actual_outcome - counterfactual_outcome)
    
    def get_loss_sensitivity(self) -> float:
        """Get the loss sensitivity coefficient"""
        return self.loss_coefficient
    
    def get_description(self) -> str:
        """Get description of loss aversion bias"""
        return f"""
        Loss aversion bias with parameters:
        - Loss coefficient (λ): {self.loss_coefficient} (losses hurt this much more than gains)
        - Gain exponent (α): {self.gain_exponent} (diminishing sensitivity to gains)
        - Loss exponent (β): {self.loss_exponent} (diminishing sensitivity to losses)
        
        This means a loss of $10 feels like losing ${self.loss_coefficient * 10:.2f} subjectively.
        """