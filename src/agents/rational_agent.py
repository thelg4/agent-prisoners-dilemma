from typing import Dict, Any, List
import json
from .base_agent import BaseAgent, Decision, Move, ReasoningStep
from datetime import datetime

class RationalAgent(BaseAgent):
    """Rational agent that maximizes expected utility without bias"""
    
    def __init__(self, agent_id: str, llm_client: Any, bias_config: Optional[Dict] = None):
        super().__init__(agent_id, llm_client, bias_config)
        self.agent_type = "rational"
    
    def decide(self, game_state: Dict[str, Any]) -> Decision:
        """Make a rational decision based on expected utility"""
        reasoning_chain = []
        
        # Step 1: Analyze situation
        situation_analysis = self._analyze_situation(game_state)
        reasoning_chain.append(ReasoningStep(
            step_type="situation_analysis",
            content=situation_analysis,
            confidence=0.9,
            timestamp=datetime.now()
        ))
        
        # Step 2: Retrieve relevant memories
        memory_analysis = self._analyze_memories()
        reasoning_chain.append(ReasoningStep(
            step_type="memory_analysis",
            content=memory_analysis,
            confidence=0.8,
            timestamp=datetime.now()
        ))
        
        # Step 3: Calculate expected utilities
        utility_analysis = self._calculate_utilities(game_state)
        reasoning_chain.append(ReasoningStep(
            step_type="utility_calculation",
            content=utility_analysis["reasoning"],
            confidence=0.95,
            timestamp=datetime.now()
        ))
        
        # Step 4: Make final decision
        final_decision = self._make_final_decision(utility_analysis)
        reasoning_chain.append(ReasoningStep(
            step_type="final_decision",
            content=final_decision["reasoning"],
            confidence=final_decision["confidence"],
            timestamp=datetime.now()
        ))
        
        return Decision(
            move=final_decision["move"],
            confidence=final_decision["confidence"],
            reasoning_chain=reasoning_chain,
            expected_payoff=final_decision["expected_payoff"],
            emotional_state="neutral"
        )
    
    def _analyze_situation(self, game_state: Dict[str, Any]) -> str:
        """Analyze the current game situation"""
        round_num = game_state.get("round_number", 0)
        total_rounds = game_state.get("total_rounds", 1000)
        payoff_matrix = game_state.get("payoff_matrix", {})
        
        prompt = f"""
        Analyze this prisoner's dilemma situation objectively:
        
        Round: {round_num}/{total_rounds}
        Payoff Matrix: {payoff_matrix}
        My current score: {self.total_score}
        My cooperation rate so far: {self.get_cooperation_rate():.2%}
        
        Provide a brief, rational analysis of the strategic situation.
        Focus on game theory principles and expected outcomes.
        """
        
        response = self.llm_client.invoke(prompt)
        return response.content
    
    def _analyze_memories(self) -> str:
        """Analyze recent memory patterns"""
        recent_memories = self.get_recent_memories(10)
        
        if not recent_memories:
            return "No previous interactions to analyze."
        
        opponent_cooperations = sum(1 for m in recent_memories if m.opponent_move == Move.COOPERATE)
        opponent_coop_rate = opponent_cooperations / len(recent_memories)
        
        my_avg_payoff = sum(m.my_payoff for m in recent_memories) / len(recent_memories)
        
        prompt = f"""
        Analyze these recent game outcomes rationally:
        
        Last {len(recent_memories)} rounds:
        - Opponent cooperation rate: {opponent_coop_rate:.2%}
        - My average payoff: {my_avg_payoff:.2f}
        - Pattern of opponent moves: {[m.opponent_move.value for m in recent_memories[-5:]]}
        
        What does this reveal about optimal strategy going forward?
        Focus on identifying patterns and expected utility implications.
        """
        
        response = self.llm_client.invoke(prompt)
        return response.content
    
    def _calculate_utilities(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate expected utilities for each move"""
        payoff_matrix = game_state.get("payoff_matrix", {})
        recent_memories = self.get_recent_memories(10)
        
        # Estimate opponent cooperation probability
        if recent_memories:
            opponent_coop_prob = sum(1 for m in recent_memories if m.opponent_move == Move.COOPERATE) / len(recent_memories)
        else:
            opponent_coop_prob = 0.5  # Default assumption
        
        # Calculate expected utilities
        cooperate_utility = (
            opponent_coop_prob * payoff_matrix.get(("cooperate", "cooperate"), 3) +
            (1 - opponent_coop_prob) * payoff_matrix.get(("cooperate", "defect"), 0)
        )
        
        defect_utility = (
            opponent_coop_prob * payoff_matrix.get(("defect", "cooperate"), 5) +
            (1 - opponent_coop_prob) * payoff_matrix.get(("defect", "defect"), 1)
        )
        
        prompt = f"""
        Calculate expected utilities:
        
        Estimated opponent cooperation probability: {opponent_coop_prob:.2%}
        Expected utility of cooperation: {cooperate_utility:.2f}
        Expected utility of defection: {defect_utility:.2f}
        
        Which move has higher expected utility and why?
        """
        
        response = self.llm_client.invoke(prompt)
        
        return {
            "cooperate_utility": cooperate_utility,
            "defect_utility": defect_utility,
            "reasoning": response.content,
            "opponent_coop_prob": opponent_coop_prob
        }
    
    def _make_final_decision(self, utility_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make the final rational decision"""
        cooperate_utility = utility_analysis["cooperate_utility"]
        defect_utility = utility_analysis["defect_utility"]
        
        if cooperate_utility > defect_utility:
            move = Move.COOPERATE
            expected_payoff = cooperate_utility
            confidence = min(0.95, 0.5 + abs(cooperate_utility - defect_utility) / 10)
        else:
            move = Move.DEFECT
            expected_payoff = defect_utility
            confidence = min(0.95, 0.5 + abs(defect_utility - cooperate_utility) / 10)
        
        reasoning = f"""
        Rational decision: {move.value}
        Expected payoff: {expected_payoff:.2f}
        Confidence: {confidence:.2%}
        
        This maximizes expected utility based on calculated probabilities.
        """
        
        return {
            "move": move,
            "expected_payoff": expected_payoff,
            "confidence": confidence,
            "reasoning": reasoning
        }