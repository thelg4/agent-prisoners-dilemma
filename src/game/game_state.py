from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from ..agents.base_agent import Move, Memory

@dataclass
class RoundResult:
    """Result of a single round of play"""
    round_number: int
    agent1_move: Move
    agent2_move: Move
    agent1_payoff: float
    agent2_payoff: float
    agent1_reasoning: str
    agent2_reasoning: str
    agent1_confidence: float
    agent2_confidence: float
    agent1_emotional_state: str
    agent2_emotional_state: str
    timestamp: datetime

@dataclass
class GameState:
    """Complete state of the game for LangGraph workflows"""
    round_number: int
    total_rounds: int
    agent1_id: str
    agent2_id: str
    agent1_score: float
    agent2_score: float
    payoff_matrix: Dict[str, Any]
    round_history: List[RoundResult]
    agent1_memories: List[Memory]
    agent2_memories: List[Memory]
    current_decision_agent1: Optional[Any] = None
    current_decision_agent2: Optional[Any] = None
    tournament_id: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}