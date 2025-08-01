from typing import Dict, Any, List
from .base_agent import BaseAgent, Decision, Move, ReasoningStep
from ..biases.loss_aversion import LossAversion
from datetime import datetime
import statistics

class LossAverseAgent(BaseAgent):
    """Agent that exhibits loss aversion bias in decision-making"""
    
    def __init__(self, agent_id: str, llm_client: Any, bias_config: Optional[Dict] = None):
        super().__init__(agent_id, llm_client, bias_config)
        self.agent_type = "loss_averse"
        self.loss_aversion = LossAversion(bias_config or {})
        self.reference_point = 0.0  # Running reference point
        
    def decide(self, game_state: Dict[str, Any]) -> Decision:
        """Make a decision influenced by loss aversion"""
        reasoning_chain = []
        
        # Update reference point
        self._update_reference_point()
        
        # Step 1: Emotional state assessment
        emotional_analysis = self._assess_emotional_state()
        reasoning_chain.append(ReasoningStep(
            step_type="emotional_assessment",
            content=emotional_analysis,
            confidence=0.8,
            timestamp=datetime.now()
        ))
        
        # Step 2: Loss-focused memory retrieval
        memory_analysis = self._analyze_painful_memories()
        reasoning_chain.append(ReasoningStep(
            step_type="loss_focused_memory",
            content=memory_analysis,
            confidence=0.9,
            timestamp=datetime.now()
        ))
        
        # Step 3: Biased utility calculation
        utility_analysis = self._calculate_biased_utilities(game_state)
        reasoning_chain.append(ReasoningStep(
            step_type="biased_utility_calculation",
            content=utility_analysis["reasoning"],
            confidence=0.85,
            timestamp=datetime.now()
        ))
        
        # Step 4: Risk assessment
        risk_analysis = self._assess_risks(game_state)
        reasoning_chain.append(ReasoningStep(
            step_type="risk_assessment",
            content=risk_analysis,
            confidence=0.8,
            timestamp=datetime.now()
        ))
        
        # Step 5: Final biased decision
        final_decision = self._make_biased_decision(utility_analysis, risk_analysis)
        reasoning_chain.append(ReasoningStep(
            step_type="biased_final_decision",
            content=final_decision["reasoning"],
            confidence=final_decision["confidence"],
            timestamp=datetime.now()
        ))
        
        return Decision(
            move=final_decision["move"],
            confidence=final_decision["confidence"],
            reasoning_chain=reasoning_chain,
            expected_payoff=final_decision["expected_payoff"],
            emotional_state=final_decision["emotional_state"]
        )
    
    def _update_reference_point(self) -> None:
        """Update the reference point for loss aversion calculations"""
        if len(self.memories) < 5:
            self.reference_point = 0.0
        else:
            # Use moving average of recent payoffs
            recent_payoffs = [m.my_payoff for m in self.memories[-10:]]
            self.reference_point = statistics.mean(recent_payoffs)
    
    def _assess_emotional_state(self) -> str:
        """Assess current emotional state based on recent outcomes"""
        recent_memories = self.get_recent_memories(5)
        
        if not recent_memories:
            emotional_state = "cautious"
            prompt = f"""
            I'm about to start playing. I feel {emotional_state} because I don't know 
            what to expect yet. The fear of being taken advantage of is already 
            weighing on my mind.
            """
        else:
            # Calculate recent pain vs pleasure
            recent_pain = sum(max(0, self.reference_point - m.my_payoff) for m in recent_memories)
            recent_pleasure = sum(max(0, m.my_payoff - self.reference_point) for m in recent_memories)
            
            if recent_pain > recent_pleasure * 1.5:  # Loss aversion factor
                emotional_state = "burned"
            elif recent_pleasure > recent_pain:
                emotional_state = "cautiously_optimistic"
            else:
                emotional_state = "wary"
            
            prompt = f"""
            Assess my emotional state after these recent outcomes:
            Recent rounds: {[(m.my_move.value, m.opponent_move.value, m.my_payoff) for m in recent_memories]}
            Reference point: {self.reference_point:.2f}
            Recent pain experienced: {recent_pain:.2f}
            Recent pleasure experienced: {recent_pleasure:.2f}
            
            I'm feeling {emotional_state}. Describe this emotional state and how it 
            affects my risk perception. Remember, losses hurt more than equivalent gains feel good.
            """
        
        response = self.llm_client.invoke(prompt)
        return response.content
    
    def _analyze_painful_memories(self) -> str:
        """Focus on painful memories (loss aversion bias)"""
        painful_memories = [
            m for m in self.memories 
            if m.my_payoff < self.reference_point and m.emotional_impact < -0.5
        ]
        
        if not painful_memories:
            return "No particularly painful memories yet, but I'm already dreading potential betrayals."
        
        # Focus disproportionately on losses
        worst_betrayal = min(painful_memories, key=lambda m: m.my_payoff)
        
        prompt = f"""
        Analyze these painful memories, focusing on the worst experiences:
        
        Worst betrayal: Round {worst_betrayal.round_number}
        - I chose: {worst_betrayal.my_move.value}
        - They chose: {worst_betrayal.opponent_move.value}  
        - My payoff: {worst_betrayal.my_payoff}
        - Emotional impact: {worst_betrayal.emotional_impact}
        
        Total painful memories: {len(painful_memories)}
        
        How do these losses still sting? What do they teach me about trust?
        Focus on the pain and regret - losses hurt more than gains feel good.
        """
        
        response = self.llm_client.invoke(prompt)
        return response.content
    
    def _calculate_biased_utilities(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate utilities with loss aversion bias applied"""
        payoff_matrix = game_state.get("payoff_matrix", {})
        recent_memories = self.get_recent_memories(10)
        
        # Estimate opponent cooperation probability
        if recent_memories:
            opponent_coop_prob = sum(1 for m in recent_memories if m.opponent_move == Move.COOPERATE) / len(recent_memories)
        else:
            opponent_coop_prob = 0.5
        
        # Calculate raw expected utilities
        cooperate_utility_raw = (
            opponent_coop_prob * payoff_matrix.get(("cooperate", "cooperate"), 3) +
            (1 - opponent_coop_prob) * payoff_matrix.get(("cooperate", "defect"), 0)
        )
        
        defect_utility_raw = (
            opponent_coop_prob * payoff_matrix.get(("defect", "cooperate"), 5) +
            (1 - opponent_coop_prob) * payoff_matrix.get(("defect", "defect"), 1)
        )
        
        # Apply loss aversion bias
        cooperate_utility_biased = self.loss_aversion.apply_bias(cooperate_utility_raw, self.reference_point)
        defect_utility_biased = self.loss_aversion.apply_bias(defect_utility_raw, self.reference_point)
        
        prompt = f"""
        Calculate utilities through my loss-averse lens:
        
        Reference point (what I expect): {self.reference_point:.2f}
        Opponent cooperation probability: {opponent_coop_prob:.2%}
        
        Raw expected utilities:
        - Cooperate: {cooperate_utility_raw:.2f}
        - Defect: {defect_utility_raw:.2f}
        
        Loss-averse adjusted utilities:
        - Cooperate: {cooperate_utility_biased:.2f} 
        - Defect: {defect_utility_biased:.2f}
        
        How does my fear of loss change these calculations? 
        Remember: losing hurts {self.loss_aversion.loss_coefficient}x more than winning feels good.
        """
        
        response = self.llm_client.invoke(prompt)
        
        return {
            "cooperate_utility_raw": cooperate_utility_raw,
            "defect_utility_raw": defect_utility_raw,
            "cooperate_utility_biased": cooperate_utility_biased,
            "defect_utility_biased": defect_utility_biased,
            "reasoning": response.content,
            "opponent_coop_prob": opponent_coop_prob
        }
    
    def _assess_risks(self, game_state: Dict[str, Any]) -> str:
        """Assess risks with loss aversion making losses loom larger"""
        round_num = game_state.get("round_number", 0)
        total_rounds = game_state.get("total_rounds", 1000)
        
        # Calculate potential for regret
        cooperate_risk = "If I cooperate and they defect, I'll get the sucker's payoff - that would really hurt"  
        defect_risk = "If I defect, I might miss out on mutual cooperation, but at least I won't be exploited"
        
        prompt = f"""
        Assess the risks of each move through my loss-averse perspective:
        
        Round {round_num}/{total_rounds}
        Current reference point: {self.reference_point:.2f}
        
        Risk of cooperation: {cooperate_risk}
        Risk of defection: {defect_risk}
        
        Which risk feels more threatening? Remember, I feel losses much more intensely 
        than equivalent gains. The pain of being betrayed looms larger than the 
        pleasure of mutual cooperation.
        """
        
        response = self.llm_client.invoke(prompt)
        return response.content
    
    def _make_biased_decision(self, utility_analysis: Dict[str, Any], risk_analysis: str) -> Dict[str, Any]:
        """Make final decision with loss aversion bias"""
        cooperate_utility = utility_analysis["cooperate_utility_biased"]
        defect_utility = utility_analysis["defect_utility_biased"]
        
        # Loss averse agents are more sensitive to potential losses
        if cooperate_utility > defect_utility:
            move = Move.COOPERATE
            expected_payoff = utility_analysis["cooperate_utility_raw"]
            confidence = min(0.9, 0.4 + abs(cooperate_utility - defect_utility) / 10)
            emotional_state = "cautiously_hopeful"
        else:
            move = Move.DEFECT
            expected_payoff = utility_analysis["defect_utility_raw"]  
            confidence = min(0.95, 0.6 + abs(defect_utility - cooperate_utility) / 10)
            emotional_state = "defensively_protected"
        
        reasoning = f"""
        Loss-averse decision: {move.value}
        
        The pain of potential loss weighs heavily on my mind. 
        Biased expected value: {max(cooperate_utility, defect_utility):.2f}
        Confidence: {confidence:.2%}
        
        I'm choosing {move.value} because {
            "I believe the potential for mutual gain outweighs the risk of betrayal" 
            if move == Move.COOPERATE 
            else "I cannot bear the thought of being exploited again"
        }.
        
        The fear of loss is driving this decision more than the hope of gain.
        """
        
        return {
            "move": move,
            "expected_payoff": expected_payoff,
            "confidence": confidence,
            "reasoning": reasoning,
            "emotional_state": emotional_state
        }