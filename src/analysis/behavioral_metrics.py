from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from ..game.tournament import TournamentResult
from ..agents.base_agent import Move

@dataclass
class BehavioralProfile:
    """Behavioral profile of an agent across tournaments"""
    agent_id: str
    agent_type: str
    avg_cooperation_rate: float
    cooperation_variance: float
    avg_final_score: float
    score_variance: float
    adaptation_rate: float
    consistency_score: float
    risk_preference: str
    strategy_description: str

class BehavioralMetrics:
    """Calculate behavioral metrics from tournament data"""
    
    def __init__(self):
        pass
    
    def calculate_cooperation_evolution(
        self, 
        cooperation_by_round: List[Tuple[bool, bool]], 
        window_size: int = 50
    ) -> List[float]:
        """Calculate cooperation rate evolution using moving average"""
        
        agent1_cooperations = [1 if coop[0] else 0 for coop in cooperation_by_round]
        
        evolution = []
        for i in range(len(agent1_cooperations)):
            start_idx = max(0, i - window_size + 1)
            end_idx = i + 1
            window_coop_rate = sum(agent1_cooperations[start_idx:end_idx]) / (end_idx - start_idx)
            evolution.append(window_coop_rate)
        
        return evolution
    
    def calculate_adaptation_rate(self, cooperation_evolution: List[float]) -> float:
        """Calculate how quickly an agent adapts its strategy"""
        
        if len(cooperation_evolution) < 100:
            return 0.0
        
        # Compare early vs late cooperation rates
        early_rate = np.mean(cooperation_evolution[:100])
        late_rate = np.mean(cooperation_evolution[-100:])
        
        # Calculate rate of change
        adaptation_rate = abs(late_rate - early_rate) / len(cooperation_evolution) * 1000
        
        return adaptation_rate
    
    def calculate_consistency_score(self, cooperation_by_round: List[Tuple[bool, bool]]) -> float:
        """Calculate strategy consistency (lower variance = more consistent)"""
        
        agent_cooperations = [1 if coop[0] else 0 for coop in cooperation_by_round]
        
        if len(agent_cooperations) < 10:
            return 0.0
        
        # Calculate variance in cooperation decisions
        variance = np.var(agent_cooperations)
        
        # Convert to consistency score (1 - normalized variance)
        consistency = 1 - (variance / 0.25)  # 0.25 is max variance for binary choices
        
        return max(0, consistency)
    
    def identify_strategy_pattern(
        self, 
        cooperation_by_round: List[Tuple[bool, bool]]
    ) -> Dict[str, Any]:
        """Identify the agent's strategic pattern"""
        
        if len(cooperation_by_round) < 20:
            return {"pattern": "insufficient_data", "confidence": 0.0}
        
        agent_moves = [coop[0] for coop in cooperation_by_round]
        opponent_moves = [coop[1] for coop in cooperation_by_round]
        
        total_rounds = len(agent_moves)
        cooperation_rate = sum(agent_moves) / total_rounds
        
        # Analyze reciprocity
        reciprocity_score = self._calculate_reciprocity(agent_moves, opponent_moves)
        
        # Identify pattern
        if cooperation_rate > 0.9:
            pattern = "always_cooperate"
            confidence = cooperation_rate
        elif cooperation_rate < 0.1:
            pattern = "always_defect"
            confidence = 1 - cooperation_rate
        elif reciprocity_score > 0.7:
            pattern = "tit_for_tat"
            confidence = reciprocity_score
        elif reciprocity_score < -0.3:
            pattern = "contrarian"
            confidence = abs(reciprocity_score)
        else:
            pattern = "mixed_strategy"
            confidence = 0.5
        
        return {
            "pattern": pattern,
            "confidence": confidence,
            "cooperation_rate": cooperation_rate,
            "reciprocity_score": reciprocity_score
        }
    
    def _calculate_reciprocity(self, agent_moves: List[bool], opponent_moves: List[bool]) -> float:
        """Calculate reciprocity score (-1 to 1)"""
        
        if len(agent_moves) < 2:
            return 0.0
        
        # Look at agent's response to opponent's previous move
        matching_responses = 0
        total_responses = 0
        
        for i in range(1, len(agent_moves)):
            prev_opponent_move = opponent_moves[i-1]
            agent_response = agent_moves[i]
            
            if agent_response == prev_opponent_move:
                matching_responses += 1
            
            total_responses += 1
        
        if total_responses == 0:
            return 0.0
        
        # Convert to score from -1 to 1
        reciprocity = (matching_responses / total_responses) * 2 - 1
        
        return reciprocity
    
    def calculate_regret_intensity(
        self, 
        tournament_result: TournamentResult
    ) -> Dict[str, float]:
        """Calculate regret intensity for both agents"""
        
        cooperation_by_round = tournament_result.cooperation_by_round
        
        agent1_regret = 0.0
        agent2_regret = 0.0
        
        for i, (a1_coop, a2_coop) in enumerate(cooperation_by_round):
            # Calculate what each agent could have earned with opposite move
            if a1_coop and not a2_coop:  # Agent 1 cooperated, agent 2 defected
                agent1_regret += 4  # Could have gotten 1 instead of 0 if defected
            elif not a1_coop and a2_coop:  # Agent 1 defected, agent 2 cooperated  
                agent1_regret += 2  # Could have gotten 3 instead of 5 if cooperated (moral regret)
            
            if a2_coop and not a1_coop:  # Agent 2 cooperated, agent 1 defected
                agent2_regret += 4  # Could have gotten 1 instead of 0 if defected
            elif not a2_coop and a1_coop:  # Agent 2 defected, agent 1 cooperated
                agent2_regret += 2  # Could have gotten 3 instead of 5 if cooperated (moral regret)
        
        return {
            "agent1_regret": agent1_regret / len(cooperation_by_round),
            "agent2_regret": agent2_regret / len(cooperation_by_round)
        }
    
    def create_behavioral_profile(
        self, 
        tournament_results: List[TournamentResult],
        agent_id: str
    ) -> BehavioralProfile:
        """Create comprehensive behavioral profile from multiple tournaments"""
        
        if not tournament_results:
            raise ValueError("No tournament results provided")
        
        # Filter results for this agent
        agent_results = []
        for result in tournament_results:
            if result.agent1_id == agent_id:
                agent_results.append({
                    "cooperation_rate": result.agent1_cooperation_rate,
                    "final_score": result.agent1_final_score,
                    "cooperation_by_round": [(coop[0], coop[1]) for coop in result.cooperation_by_round],
                    "agent_type": result.agent1_type
                })
            elif result.agent2_id == agent_id:
                agent_results.append({
                    "cooperation_rate": result.agent2_cooperation_rate,
                    "final_score": result.agent2_final_score,
                    "cooperation_by_round": [(coop[1], coop[0]) for coop in result.cooperation_by_round],
                    "agent_type": result.agent2_type
                })
        
        if not agent_results:
            raise ValueError(f"No results found for agent {agent_id}")
        
        # Calculate metrics
        cooperation_rates = [r["cooperation_rate"] for r in agent_results]
        final_scores = [r["final_score"] for r in agent_results]
        
        avg_cooperation_rate = np.mean(cooperation_rates)
        cooperation_variance = np.var(cooperation_rates)
        avg_final_score = np.mean(final_scores)
        score_variance = np.var(final_scores)
        
        # Calculate adaptation rate (average across tournaments)
        adaptation_rates = []
        consistency_scores = []
        
        for result in agent_results:
            evolution = self.calculate_cooperation_evolution(result["cooperation_by_round"])
            adaptation_rates.append(self.calculate_adaptation_rate(evolution))
            consistency_scores.append(self.calculate_consistency_score(result["cooperation_by_round"]))
        
        avg_adaptation_rate = np.mean(adaptation_rates)
        avg_consistency_score = np.mean(consistency_scores)
        
        # Determine risk preference
        if avg_cooperation_rate > 0.7:
            risk_preference = "risk_seeking"  # Willing to cooperate despite betrayal risk
        elif avg_cooperation_rate < 0.3:
            risk_preference = "risk_averse"  # Avoids cooperation to prevent exploitation
        else:
            risk_preference = "risk_neutral"
        
        # Generate strategy description
        strategy_patterns = [self.identify_strategy_pattern(r["cooperation_by_round"]) 
                           for r in agent_results]
        most_common_pattern = max(set(p["pattern"] for p in strategy_patterns), 
                                key=[p["pattern"] for p in strategy_patterns].count)
        
        strategy_description = f"Primarily uses {most_common_pattern} strategy with {avg_cooperation_rate:.1%} cooperation rate"
        
        return BehavioralProfile(
            agent_id=agent_id,
            agent_type=agent_results[0]["agent_type"],
            avg_cooperation_rate=avg_cooperation_rate,
            cooperation_variance=cooperation_variance,
            avg_final_score=avg_final_score,
            score_variance=score_variance,
            adaptation_rate=avg_adaptation_rate,
            consistency_score=avg_consistency_score,
            risk_preference=risk_preference,
            strategy_description=strategy_description
        )