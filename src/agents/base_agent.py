from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class Move(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"

@dataclass
class Memory:
    """Represents a memory of a past game round"""
    round_number: int
    my_move: Move
    opponent_move: Move
    my_payoff: float
    opponent_payoff: float
    emotional_impact: float  # How much this round "hurt" or "pleased"
    timestamp: datetime

@dataclass
class ReasoningStep:
    """One step in an agent's reasoning chain"""
    step_type: str  # "memory_retrieval", "situation_analysis", etc.
    content: str
    confidence: float
    timestamp: datetime

@dataclass
class Decision:
    """Agent's decision with supporting information"""
    move: Move
    confidence: float
    reasoning_chain: List[ReasoningStep]
    expected_payoff: float
    emotional_state: str

class BaseAgent(ABC):
    """Abstract base class for all game-playing agents"""
    
    def __init__(self, agent_id: str, llm_client: Any, bias_config: Optional[Dict] = None):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.bias_config = bias_config or {}
        self.memories: List[Memory] = []
        self.total_score = 0
        self.round_count = 0
        
    @abstractmethod
    def decide(self, game_state: Dict[str, Any]) -> Decision:
        """Make a decision for the current round"""
        pass
    
    def update_memory(self, memory: Memory) -> None:
        """Add a new memory from the completed round"""
        self.memories.append(memory)
        self.total_score += memory.my_payoff
        self.round_count += 1
    
    def get_cooperation_rate(self) -> float:
        """Calculate historical cooperation rate"""
        if not self.memories:
            return 0.0
        cooperations = sum(1 for m in self.memories if m.my_move == Move.COOPERATE)
        return cooperations / len(self.memories)
    
    def get_recent_memories(self, n: int = 10) -> List[Memory]:
        """Get the n most recent memories"""
        return self.memories[-n:] if self.memories else []