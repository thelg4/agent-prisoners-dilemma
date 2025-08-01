from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from ..agents.base_agent import Move

@dataclass
class PayoffMatrix:
    """Payoff matrix for prisoner's dilemma"""
    cooperate_cooperate: Tuple[float, float] = (3, 3)    # Mutual cooperation
    cooperate_defect: Tuple[float, float] = (0, 5)       # Sucker's payoff  
    defect_cooperate: Tuple[float, float] = (5, 0)       # Temptation payoff
    defect_defect: Tuple[float, float] = (1, 1)          # Mutual defection
    
    def get_payoffs(self, move1: Move, move2: Move) -> Tuple[float, float]:
        """Get payoffs for both players given their moves"""
        key = (move1.value, move2.value)
        
        payoff_map = {
            ("cooperate", "cooperate"): self.cooperate_cooperate,
            ("cooperate", "defect"): self.cooperate_defect,
            ("defect", "cooperate"): self.defect_cooperate,
            ("defect", "defect"): self.defect_defect
        }
        
        return payoff_map[key]
    
    def to_dict(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """Convert to dictionary format"""
        return {
            ("cooperate", "cooperate"): self.cooperate_cooperate,
            ("cooperate", "defect"): self.cooperate_defect,
            ("defect", "cooperate"): self.defect_cooperate,
            ("defect", "defect"): self.defect_defect
        }

class PrisonerDilemma:
    """Prisoner's dilemma game implementation"""
    
    def __init__(self, payoff_matrix: Optional[PayoffMatrix] = None):
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        self.round_number = 0
        self.game_history: List[Tuple[Move, Move, Tuple[float, float]]] = []
    
    def play_round(self, move1: Move, move2: Move) -> Tuple[float, float]:
        """Play a single round and return payoffs"""
        payoffs = self.payoff_matrix.get_payoffs(move1, move2)
        self.game_history.append((move1, move2, payoffs))
        self.round_number += 1
        return payoffs
    
    def get_game_state(self, total_rounds: int) -> Dict[str, Any]:
        """Get current game state for agents"""
        return {
            "round_number": self.round_number,
            "total_rounds": total_rounds,
            "payoff_matrix": self.payoff_matrix.to_dict(),
            "history": self.game_history.copy()
        }
    
    def reset(self) -> None:
        """Reset the game to initial state"""
        self.round_number = 0
        self.game_history.clear()