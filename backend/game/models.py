from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime


class Move(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"


class AgentType(Enum):
    ECON = "econ"      # Rational economic agent
    HUMAN = "human"    # Loss-averse behavioral agent


@dataclass
class GameResult:
    """Single round result"""
    agent1_move: Move
    agent2_move: Move
    agent1_payoff: float
    agent2_payoff: float
    agent1_adjusted_payoff: float  # After loss aversion
    agent2_adjusted_payoff: float  # After loss aversion
    round_number: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Individual agent state and statistics"""
    agent_id: str
    agent_type: AgentType
    name: str
    total_payoff: float = 0.0
    adjusted_total_payoff: float = 0.0  # After loss aversion adjustments
    moves_history: List[Move] = field(default_factory=list)
    opponent_moves_history: List[Move] = field(default_factory=list)
    cooperation_rate: float = 0.0
    defection_rate: float = 0.0
    average_payoff_per_round: float = 0.0


class GameState(TypedDict):
    """Overall game state for LangGraph"""
    # Game configuration
    game_id: str
    max_rounds: int
    current_round: int
    loss_aversion_factor: float
    reference_point: float  # Point from which losses are calculated
    
    # Agent states
    agent1: AgentState
    agent2: AgentState
    
    # Game history and results
    game_history: List[GameResult]
    current_moves: Dict[str, Optional[Move]]
    
    # Game status
    game_complete: bool
    winner: Optional[str]
    game_summary: Optional[Dict[str, Any]]
    
    # Metadata
    created_at: datetime
    updated_at: datetime


@dataclass
class GameConfig:
    """Configuration for creating a new game"""
    max_rounds: int = 10
    loss_aversion_factor: float = 2.0
    agent1_type: AgentType = AgentType.ECON
    agent2_type: AgentType = AgentType.HUMAN
    agent1_name: str = "Econ Agent"
    agent2_name: str = "Human Agent"
    reference_point: float = 2.5