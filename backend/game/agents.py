import random
import math
from typing import List
from .models import Move, AgentType, GameState, AgentState
from .payoffs import PayoffMatrix


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
    
    def decide_move(self, state: GameState, agent_state: AgentState) -> Move:
        """Override in subclasses"""
        raise NotImplementedError
    
    def _get_opponent_cooperation_rate(self, agent_state: AgentState) -> float:
        """Calculate opponent's historical cooperation rate"""
        if not agent_state.opponent_moves_history:
            return 0.5  # Assume neutral if no history
        
        cooperations = agent_state.opponent_moves_history.count(Move.COOPERATE)
        return cooperations / len(agent_state.opponent_moves_history)


class EconAgent(BaseAgent):
    """Rational economic agent - pure game theory approach"""
    
    def __init__(self, agent_id: str, name: str = "Econ Agent"):
        super().__init__(agent_id, name)
        self.agent_type = AgentType.ECON
    
    def decide_move(self, state: GameState, agent_state: AgentState) -> Move:
        """
        Rational strategy based on game theory principles.
        Uses a combination of:
        1. Tit-for-tat with forgiveness
        2. Exploitation of overly cooperative opponents
        3. Mixed strategies to avoid being predictable
        """
        # First round: cooperate to test opponent (generous start)
        if len(agent_state.moves_history) == 0:
            return Move.COOPERATE if random.random() < 0.6 else Move.DEFECT
        
        opponent_coop_rate = self._get_opponent_cooperation_rate(agent_state)
        rounds_played = len(agent_state.moves_history)
        
        # Early game (first 3 rounds): more exploratory
        if rounds_played <= 3:
            return self._early_game_strategy(agent_state, opponent_coop_rate)
        
        # Mid to late game: more strategic
        return self._strategic_play(agent_state, opponent_coop_rate, state)
    
    def _early_game_strategy(self, agent_state: AgentState, opponent_coop_rate: float) -> Move:
        """Strategy for the first few rounds"""
        # Mostly tit-for-tat with some exploration
        last_opponent_move = agent_state.opponent_moves_history[-1]
        
        if last_opponent_move == Move.COOPERATE:
            # Respond to cooperation with cooperation (90% of time)
            return Move.COOPERATE if random.random() < 0.9 else Move.DEFECT
        else:
            # Respond to defection with defection (80% of time)
            return Move.DEFECT if random.random() < 0.8 else Move.COOPERATE
    
    def _strategic_play(self, agent_state: AgentState, opponent_coop_rate: float, state: GameState) -> Move:
        """Strategic play based on opponent patterns"""
        
        # If opponent is very cooperative (>70%), exploit them more
        if opponent_coop_rate > 0.7:
            return Move.DEFECT if random.random() < 0.8 else Move.COOPERATE
        
        # If opponent rarely cooperates (<30%), mostly defect
        elif opponent_coop_rate < 0.3:
            return Move.DEFECT if random.random() < 0.9 else Move.COOPERATE
        
        # For balanced opponents, use advanced strategies
        else:
            return self._advanced_strategy(agent_state, opponent_coop_rate, state)
    
    def _advanced_strategy(self, agent_state: AgentState, opponent_coop_rate: float, state: GameState) -> Move:
        """Advanced strategies for balanced opponents"""
        
        # Look for patterns in opponent's recent moves
        recent_moves = agent_state.opponent_moves_history[-5:] if len(agent_state.opponent_moves_history) >= 5 else agent_state.opponent_moves_history
        
        # If opponent is alternating, try to break the pattern
        if len(recent_moves) >= 4 and self._is_alternating(recent_moves):
            return Move.DEFECT
        
        # Calculate expected payoffs for both moves
        coop_expected = PayoffMatrix.calculate_expected_payoff(Move.COOPERATE, opponent_coop_rate)
        defect_expected = PayoffMatrix.calculate_expected_payoff(Move.DEFECT, opponent_coop_rate)
        
        # Add small random factor to avoid being too predictable
        noise = random.gauss(0, 0.1)
        coop_expected += noise
        
        return Move.COOPERATE if coop_expected > defect_expected else Move.DEFECT
    
    def _is_alternating(self, moves: List[Move]) -> bool:
        """Check if moves are alternating between cooperate and defect"""
        if len(moves) < 4:
            return False
        
        for i in range(len(moves) - 1):
            if moves[i] == moves[i + 1]:
                return False
        return True


class HumanAgent(BaseAgent):
    """Loss-averse behavioral agent - feels pain of losses more strongly"""
    
    def __init__(self, agent_id: str, name: str = "Human Agent"):
        super().__init__(agent_id, name)
        self.agent_type = AgentType.HUMAN
        self.trust_level = 0.7  # Initial trust level
        self.fear_factor = 1.0   # How much we fear being exploited
    
    def decide_move(self, state: GameState, agent_state: AgentState) -> Move:
        """
        Decision-making influenced by loss aversion and emotional factors.
        The "sucker's payoff" feels especially painful.
        """
        loss_factor = state["loss_aversion_factor"]
        reference_point = state["reference_point"]
        
        # First round: optimistic start (humans tend to be initially trusting)
        if len(agent_state.moves_history) == 0:
            return Move.COOPERATE if random.random() < 0.7 else Move.DEFECT
        
        # Update trust and fear based on recent experiences
        self._update_emotional_state(agent_state, loss_factor, reference_point)
        
        # Calculate expected utility with loss aversion
        opponent_coop_rate = self._estimate_opponent_cooperation_probability(agent_state)
        
        coop_utility = self._calculate_expected_utility(
            Move.COOPERATE, opponent_coop_rate, loss_factor, reference_point
        )
        
        defect_utility = self._calculate_expected_utility(
            Move.DEFECT, opponent_coop_rate, loss_factor, reference_point
        )
        
        # Add emotional noise and bias
        coop_utility += self._get_emotional_bias(Move.COOPERATE, agent_state)
        defect_utility += self._get_emotional_bias(Move.DEFECT, agent_state)
        
        # Make decision with some randomness (humans aren't perfectly rational)
        decision_noise = random.gauss(0, 0.15)
        coop_utility += decision_noise
        
        return Move.COOPERATE if coop_utility > defect_utility else Move.DEFECT
    
    def _update_emotional_state(self, agent_state: AgentState, loss_factor: float, reference_point: float):
        """Update trust and fear based on recent experiences"""
        if not agent_state.moves_history:
            return
        
        # Look at last few rounds to update emotional state
        recent_rounds = min(3, len(agent_state.moves_history))
        recent_my_moves = agent_state.moves_history[-recent_rounds:]
        recent_opp_moves = agent_state.opponent_moves_history[-recent_rounds:]
        
        # Update trust based on opponent's responses to our cooperation
        trust_change = 0
        fear_change = 0
        
        for my_move, opp_move in zip(recent_my_moves, recent_opp_moves):
            if my_move == Move.COOPERATE and opp_move == Move.COOPERATE:
                trust_change += 0.1  # Trust increases when cooperation is reciprocated
            elif my_move == Move.COOPERATE and opp_move == Move.DEFECT:
                trust_change -= 0.2  # Trust decreases more when we're exploited
                fear_change += 0.15   # Fear of being sucker increases
        
        self.trust_level = max(0.1, min(1.0, self.trust_level + trust_change))
        self.fear_factor = max(0.5, min(2.0, self.fear_factor + fear_change))
    
    def _estimate_opponent_cooperation_probability(self, agent_state: AgentState) -> float:
        """Estimate opponent cooperation probability with recency bias"""
        if not agent_state.opponent_moves_history:
            return 0.5
        
        # Weight recent moves more heavily (recency bias)
        total_weight = 0
        weighted_cooperations = 0
        
        for i, move in enumerate(agent_state.opponent_moves_history):
            weight = 1.0 + (i * 0.1)  # More recent moves get higher weight
            total_weight += weight
            if move == Move.COOPERATE:
                weighted_cooperations += weight
        
        base_rate = weighted_cooperations / total_weight
        
        # Adjust based on trust level
        adjusted_rate = base_rate * self.trust_level + (1 - self.trust_level) * 0.3
        
        return max(0.1, min(0.9, adjusted_rate))
    
    def _calculate_expected_utility(self, my_move: Move, opponent_coop_rate: float,
                                  loss_factor: float, reference_point: float) -> float:
        """Calculate expected utility with loss aversion"""
        p_coop = opponent_coop_rate
        p_defect = 1 - p_coop
        
        # Get payoffs for each scenario
        if my_move == Move.COOPERATE:
            payoff_vs_coop = PayoffMatrix.get_payoffs(Move.COOPERATE, Move.COOPERATE)[0]
            payoff_vs_defect = PayoffMatrix.get_payoffs(Move.COOPERATE, Move.DEFECT)[0]
        else:
            payoff_vs_coop = PayoffMatrix.get_payoffs(Move.DEFECT, Move.COOPERATE)[0]
            payoff_vs_defect = PayoffMatrix.get_payoffs(Move.DEFECT, Move.DEFECT)[0]
        
        # Apply loss aversion
        adjusted_vs_coop = PayoffMatrix.apply_loss_aversion(
            payoff_vs_coop, loss_factor, reference_point
        )
        adjusted_vs_defect = PayoffMatrix.apply_loss_aversion(
            payoff_vs_defect, loss_factor, reference_point
        )
        
        # Calculate expected utility
        expected_utility = (p_coop * adjusted_vs_coop) + (p_defect * adjusted_vs_defect)
        return expected_utility
    
    def _get_emotional_bias(self, move: Move, agent_state: AgentState) -> float:
        """Add emotional bias to decision making"""
        bias = 0
        
        # Fear of being exploited makes cooperation less appealing
        if move == Move.COOPERATE:
            bias -= (self.fear_factor - 1.0) * 0.3
        
        # Trust makes cooperation more appealing
        if move == Move.COOPERATE:
            bias += (self.trust_level - 0.5) * 0.2
        
        # If we've been hurt recently, become more defensive
        if len(agent_state.moves_history) >= 2:
            last_my_move = agent_state.moves_history[-1]
            last_opp_move = agent_state.opponent_moves_history[-1]
            
            if last_my_move == Move.COOPERATE and last_opp_move == Move.DEFECT:
                if move == Move.DEFECT:
                    bias += 0.25  # Bias toward defection after being exploited
        
        return bias


def create_agent(agent_id: str, agent_type: AgentType, name: str) -> BaseAgent:
    """Factory function to create agents"""
    if agent_type == AgentType.ECON:
        return EconAgent(agent_id, name)
    elif agent_type == AgentType.HUMAN:
        return HumanAgent(agent_id, name)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")