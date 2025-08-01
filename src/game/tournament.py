from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from ..agents.base_agent import BaseAgent, Memory, Move
from .prisoner_dilemma import PrisonerDilemma, PayoffMatrix
from .game_state import RoundResult

@dataclass
class TournamentResult:
    """Results from a tournament between agents"""
    tournament_id: str
    agent1_id: str
    agent2_id: str
    agent1_type: str
    agent2_type: str
    total_rounds: int
    agent1_final_score: float
    agent2_final_score: float
    agent1_cooperation_rate: float
    agent2_cooperation_rate: float
    cooperation_by_round: List[Tuple[bool, bool]]  # (agent1_cooperated, agent2_cooperated)
    final_payoffs: Tuple[float, float]
    tournament_duration: float  # seconds
    timestamp: datetime
    round_details: List[RoundResult] = field(default_factory=list)

class Tournament:
    """Manages tournaments between agents"""
    
    def __init__(self, tournament_id: str, payoff_matrix: Optional[PayoffMatrix] = None):
        self.tournament_id = tournament_id
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        self.logger = logging.getLogger(__name__)
    
    def run_tournament(
        self, 
        agent1: BaseAgent, 
        agent2: BaseAgent, 
        num_rounds: int = 1000
    ) -> TournamentResult:
        """Run a complete tournament between two agents"""
        
        start_time = datetime.now()
        self.logger.info(f"Starting tournament {self.tournament_id}: {agent1.agent_id} vs {agent2.agent_id}")
        
        # Initialize game
        game = PrisonerDilemma(self.payoff_matrix)
        round_details = []
        cooperation_by_round = []
        
        # Play all rounds
        for round_num in range(1, num_rounds + 1):
            # Get game state for agents
            game_state = game.get_game_state(num_rounds)
            game_state["round_number"] = round_num
            
            # Agents make decisions
            decision1 = agent1.decide(game_state)
            decision2 = agent2.decide(game_state)
            
            # Play the round
            payoffs = game.play_round(decision1.move, decision2.move)
            
            # Calculate emotional impact (for biased agents)
            emotional_impact1 = self._calculate_emotional_impact(
                agent1, decision1.move, decision2.move, payoffs[0]
            )
            emotional_impact2 = self._calculate_emotional_impact(
                agent2, decision2.move, decision1.move, payoffs[1]
            )
            
            # Create memories for agents
            memory1 = Memory(
                round_number=round_num,
                my_move=decision1.move,
                opponent_move=decision2.move,
                my_payoff=payoffs[0],
                opponent_payoff=payoffs[1],
                emotional_impact=emotional_impact1,
                timestamp=datetime.now()
            )
            
            memory2 = Memory(
                round_number=round_num,
                my_move=decision2.move,
                opponent_move=decision1.move,
                my_payoff=payoffs[1],
                opponent_payoff=payoffs[0],
                emotional_impact=emotional_impact2,
                timestamp=datetime.now()
            )
            
            # Update agent memories
            agent1.update_memory(memory1)
            agent2.update_memory(memory2)
            
            # Record round details
            round_result = RoundResult(
                round_number=round_num,
                agent1_move=decision1.move,
                agent2_move=decision2.move,
                agent1_payoff=payoffs[0],
                agent2_payoff=payoffs[1],
                agent1_reasoning=self._extract_reasoning_summary(decision1.reasoning_chain),
                agent2_reasoning=self._extract_reasoning_summary(decision2.reasoning_chain),
                agent1_confidence=decision1.confidence,
                agent2_confidence=decision2.confidence,
                agent1_emotional_state=decision1.emotional_state,
                agent2_emotional_state=decision2.emotional_state,
                timestamp=datetime.now()
            )
            
            round_details.append(round_result)
            cooperation_by_round.append((
                decision1.move == Move.COOPERATE,
                decision2.move == Move.COOPERATE
            ))
            
            # Log progress periodically
            if round_num % 100 == 0:
                self.logger.info(f"Tournament {self.tournament_id}: Round {round_num}/{num_rounds}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate final results
        result = TournamentResult(
            tournament_id=self.tournament_id,
            agent1_id=agent1.agent_id,
            agent2_id=agent2.agent_id,
            agent1_type=agent1.agent_type,
            agent2_type=agent2.agent_type,
            total_rounds=num_rounds,
            agent1_final_score=agent1.total_score,
            agent2_final_score=agent2.total_score,
            agent1_cooperation_rate=agent1.get_cooperation_rate(),
            agent2_cooperation_rate=agent2.get_cooperation_rate(),
            cooperation_by_round=cooperation_by_round,
            final_payoffs=(agent1.total_score, agent2.total_score),
            tournament_duration=duration,
            timestamp=end_time,
            round_details=round_details
        )
        
        self.logger.info(f"Tournament {self.tournament_id} completed in {duration:.2f}s")
        self.logger.info(f"Final scores: {agent1.agent_id}={agent1.total_score}, {agent2.agent_id}={agent2.total_score}")
        
        return result
    
    def _calculate_emotional_impact(
        self, 
        agent: BaseAgent, 
        my_move: Move, 
        opponent_move: Move, 
        my_payoff: float
    ) -> float:
        """Calculate emotional impact of the round outcome"""
        
        # Base emotional impact on payoff relative to expectations
        if hasattr(agent, 'reference_point'):
            reference = agent.reference_point
        else:
            reference = 2.0  # Neutral expectation
        
        raw_impact = my_payoff - reference
        
        # Amplify negative emotions for loss-averse agents
        if hasattr(agent, 'loss_aversion') and raw_impact < 0:
            return raw_impact * agent.loss_aversion.loss_coefficient
        
        return raw_impact
    
    def _extract_reasoning_summary(self, reasoning_chain: List[Any]) -> str:
        """Extract a summary from the reasoning chain"""
        if not reasoning_chain:
            return "No reasoning recorded"
        
        # Get the final decision reasoning
        final_step = reasoning_chain[-1]
        return final_step.content[:200] + "..." if len(final_step.content) > 200 else final_step.content