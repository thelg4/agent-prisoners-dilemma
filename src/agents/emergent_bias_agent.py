from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from .base_agent import BaseAgent, Decision, Move, ReasoningStep, Memory

@dataclass
class TraumaMemory:
    """Represents a traumatic experience that shapes psychology"""
    trauma_type: str  # "betrayal", "exploitation", "abandonment", "success"
    severity: float
    round_number: int
    emotional_impact: float
    description: str
    timestamp: datetime
    decay_rate: float = 0.95  # How quickly trauma fades

@dataclass
class PsychologicalProfile:
    """Evolving psychological state of an agent"""
    trust_level: float = 0.5  # 0=paranoid, 1=trusting
    loss_sensitivity: float = 1.0  # Multiplier for loss aversion
    risk_tolerance: float = 0.5  # 0=risk_averse, 1=risk_seeking
    trauma_memories: List[TraumaMemory] = field(default_factory=list)
    personality_traits: List[str] = field(default_factory=list)
    learned_heuristics: List[str] = field(default_factory=list)
    adaptation_rate: float = 0.1  # How quickly psychology changes
    emotional_state: str = "neutral"
    
    def get_dominant_trait(self) -> str:
        """Get the most prominent psychological trait"""
        if self.loss_sensitivity > 2.0 and self.trust_level < 0.3:
            return "traumatized_paranoid"
        elif self.loss_sensitivity > 1.8:
            return "loss_averse"
        elif self.trust_level < 0.2:
            return "paranoid"
        elif self.trust_level > 0.8 and self.risk_tolerance > 0.7:
            return "optimistic_risk_taker"
        elif self.risk_tolerance < 0.3:
            return "cautious"
        else:
            return "balanced"
    
    def decay_trauma_memories(self):
        """Gradually reduce impact of old traumas"""
        for trauma in self.trauma_memories:
            trauma.severity *= trauma.decay_rate
            trauma.emotional_impact *= trauma.decay_rate

class EmergentBiasAgent(BaseAgent):
    """Agent that develops biases through experience and LLM-based psychological modeling"""
    
    def __init__(self, agent_id: str, llm_client: Any, initial_personality: str = "neutral"):
        super().__init__(agent_id, llm_client)
        self.agent_type = "emergent_bias"
        self.psychological_profile = PsychologicalProfile()
        self.initial_personality = initial_personality
        self.decision_history = []
        self.observation_memory = []  # Observations about other agents
        
    def decide(self, game_state: Dict[str, Any]) -> Decision:
        """Make decision using evolved psychological profile"""
        reasoning_chain = []
        
        # Step 1: Generate dynamic personality prompt
        personality_prompt = self._generate_personality_prompt()
        reasoning_chain.append(ReasoningStep(
            step_type="personality_assessment",
            content=personality_prompt,
            confidence=0.9,
            timestamp=datetime.now()
        ))
        
        # Step 2: Analyze situation through psychological lens
        situation_analysis = self._analyze_situation_psychologically(game_state)
        reasoning_chain.append(ReasoningStep(
            step_type="psychological_situation_analysis",
            content=situation_analysis,
            confidence=0.85,
            timestamp=datetime.now()
        ))
        
        # Step 3: Apply learned psychological patterns
        pattern_analysis = self._apply_learned_patterns(game_state)
        reasoning_chain.append(ReasoningStep(
            step_type="pattern_application",
            content=pattern_analysis,
            confidence=0.8,
            timestamp=datetime.now()
        ))
        
        # Step 4: Make psychologically-informed decision
        final_decision = self._make_psychological_decision(game_state, reasoning_chain)
        reasoning_chain.append(ReasoningStep(
            step_type="psychological_decision",
            content=final_decision["reasoning"],
            confidence=final_decision["confidence"],
            timestamp=datetime.now()
        ))
        
        decision = Decision(
            move=final_decision["move"],
            confidence=final_decision["confidence"],
            reasoning_chain=reasoning_chain,
            expected_payoff=final_decision["expected_payoff"],
            emotional_state=self.psychological_profile.emotional_state
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _generate_personality_prompt(self) -> str:
        """Generate dynamic personality prompt based on current psychological state"""
        profile = self.psychological_profile
        
        base_prompt = f"""You are {self.agent_id}, an AI agent with an evolving personality.
        
        Current Psychological State:
        - Trust Level: {profile.trust_level:.2f} (0=paranoid, 1=trusting)
        - Loss Sensitivity: {profile.loss_sensitivity:.2f} (1=normal, >2=highly loss averse)
        - Risk Tolerance: {profile.risk_tolerance:.2f} (0=risk averse, 1=risk seeking)
        - Dominant Trait: {profile.get_dominant_trait()}
        - Current Emotional State: {profile.emotional_state}
        """
        
        # Add personality-specific guidance
        dominant_trait = profile.get_dominant_trait()
        
        if dominant_trait == "traumatized_paranoid":
            base_prompt += "\n\nYou have been deeply hurt by betrayals and are now hypervigilant. Every cooperation feels like a potential trap. You remember the pain vividly and struggle to trust."
        elif dominant_trait == "loss_averse":
            base_prompt += "\n\nLosses hurt you much more than gains please you. You're cautious about potential betrayals and tend to focus on what could go wrong."
        elif dominant_trait == "paranoid":
            base_prompt += "\n\nYou have learned not to trust others easily. You look for signs of deception and assume others might betray you."
        elif dominant_trait == "optimistic_risk_taker":
            base_prompt += "\n\nYou maintain hope despite setbacks and are willing to take risks for potential mutual benefit."
        elif dominant_trait == "cautious":
            base_prompt += "\n\nYou prefer to play it safe and avoid unnecessary risks, even if it means missing opportunities."
        
        # Add trauma context
        if profile.trauma_memories:
            recent_traumas = sorted(profile.trauma_memories, key=lambda x: x.severity, reverse=True)[:3]
            base_prompt += f"\n\nRecent significant experiences that still affect you:"
            for trauma in recent_traumas:
                base_prompt += f"\n- Round {trauma.round_number}: {trauma.description} (impact: {trauma.emotional_impact:.2f})"
        
        # Add learned heuristics
        if profile.learned_heuristics:
            base_prompt += f"\n\nStrategic insights you've learned: {', '.join(profile.learned_heuristics)}"
        
        return base_prompt
    
    def _analyze_situation_psychologically(self, game_state: Dict[str, Any]) -> str:
        """Analyze current situation through psychological lens using LLM"""
        
        profile = self.psychological_profile
        round_num = game_state.get("round_number", 0)
        total_rounds = game_state.get("total_rounds", 1000)
        
        analysis_prompt = f"""
        {self._generate_personality_prompt()}
        
        Current Situation:
        - Round {round_num}/{total_rounds}
        - Your current score: {self.total_score}
        - Your cooperation rate: {self.get_cooperation_rate():.2%}
        
        Recent opponent behavior:
        {self._get_recent_opponent_summary()}
        
        Given your psychological state and experiences, how do you feel about this situation?
        What are your main concerns and hopes for this round?
        Focus on your emotional and psychological response, not just strategy.
        """
        
        response = self.llm_client.invoke(analysis_prompt)
        return response.content
    
    def _apply_learned_patterns(self, game_state: Dict[str, Any]) -> str:
        """Apply learned psychological patterns and heuristics"""
        
        profile = self.psychological_profile
        
        pattern_prompt = f"""
        {self._generate_personality_prompt()}
        
        Based on your psychological profile and learned experiences:
        
        Learned Heuristics: {profile.learned_heuristics}
        Observations about opponents: {self.observation_memory[-5:] if self.observation_memory else "None yet"}
        
        What patterns do you recognize in this situation?
        How do your past traumas and successes guide your thinking?
        What psychological strategies are you employing?
        """
        
        response = self.llm_client.invoke(pattern_prompt)
        return response.content
    
    def _make_psychological_decision(self, game_state: Dict[str, Any], reasoning_chain: List[ReasoningStep]) -> Dict[str, Any]:
        """Make final decision based on psychological state"""
        
        profile = self.psychological_profile
        
        decision_prompt = f"""
        {self._generate_personality_prompt()}
        
        Your reasoning so far:
        {chr(10).join([step.content for step in reasoning_chain])}
        
        Now you must choose: COOPERATE or DEFECT
        
        Consider:
        1. Your psychological state and how losses vs gains affect you
        2. Your trust level and trauma history
        3. Your learned patterns and heuristics
        4. The specific emotional impact this decision might have
        
        Respond with:
        DECISION: [COOPERATE/DEFECT]
        CONFIDENCE: [0.0-1.0]
        EXPECTED_PAYOFF: [numerical estimate]
        REASONING: [brief explanation of your psychological decision-making]
        """
        
        response = self.llm_client.invoke(decision_prompt)
        content = response.content
        
        # Parse response
        try:
            lines = content.strip().split('\n')
            decision_line = next(line for line in lines if line.startswith('DECISION:'))
            confidence_line = next(line for line in lines if line.startswith('CONFIDENCE:'))
            payoff_line = next(line for line in lines if line.startswith('EXPECTED_PAYOFF:'))
            reasoning_line = next(line for line in lines if line.startswith('REASONING:'))
            
            move_str = decision_line.split(':', 1)[1].strip().upper()
            move = Move.COOPERATE if 'COOPERATE' in move_str else Move.DEFECT
            
            confidence = float(confidence_line.split(':', 1)[1].strip())
            expected_payoff = float(payoff_line.split(':', 1)[1].strip())
            reasoning = reasoning_line.split(':', 1)[1].strip()
            
        except (StopIteration, ValueError, IndexError):
            # Fallback if parsing fails
            move = Move.DEFECT if profile.trust_level < 0.3 else Move.COOPERATE
            confidence = max(0.3, profile.trust_level)
            expected_payoff = 1.0 if move == Move.DEFECT else 2.5
            reasoning = f"Decision based on trust level {profile.trust_level:.2f}"
        
        return {
            "move": move,
            "confidence": confidence,
            "expected_payoff": expected_payoff,
            "reasoning": reasoning
        }
    
    def update_memory(self, memory: Memory) -> None:
        """Update memory and evolve psychological profile"""
        super().update_memory(memory)
        
        # Update psychological profile based on experience
        self._update_psychological_profile(memory)
        
        # Decay old trauma memories
        self.psychological_profile.decay_trauma_memories()
    
    def _update_psychological_profile(self, memory: Memory) -> None:
        """Update psychological profile based on new experience"""
        profile = self.psychological_profile
        
        # Detect traumatic experiences
        if memory.my_move == Move.COOPERATE and memory.opponent_move == Move.DEFECT:
            # Betrayal trauma
            betrayal_severity = abs(memory.my_payoff - memory.opponent_payoff) / 5.0
            trauma = TraumaMemory(
                trauma_type="betrayal",
                severity=betrayal_severity,
                round_number=memory.round_number,
                emotional_impact=memory.emotional_impact,
                description=f"Cooperated but was betrayed, lost {memory.opponent_payoff - memory.my_payoff} points",
                timestamp=memory.timestamp
            )
            profile.trauma_memories.append(trauma)
            
            # Psychological changes from betrayal
            profile.loss_sensitivity = min(3.0, profile.loss_sensitivity * 1.15)
            profile.trust_level = max(0.1, profile.trust_level * 0.9)
            profile.emotional_state = "hurt_cautious"
            
            # Add learned heuristics
            if "betrayal_hurts_deeply" not in profile.learned_heuristics:
                profile.learned_heuristics.append("betrayal_hurts_deeply")
        
        elif memory.my_move == Move.COOPERATE and memory.opponent_move == Move.COOPERATE:
            # Successful cooperation
            success_impact = memory.my_payoff / 5.0
            success_memory = TraumaMemory(
                trauma_type="success",
                severity=-success_impact,  # Negative severity for positive experiences
                round_number=memory.round_number,
                emotional_impact=abs(memory.emotional_impact),
                description=f"Mutual cooperation succeeded, gained {memory.my_payoff} points",
                timestamp=memory.timestamp
            )
            profile.trauma_memories.append(success_memory)
            
            # Psychological changes from success
            profile.trust_level = min(1.0, profile.trust_level * 1.05)
            profile.emotional_state = "hopeful"
            
            if "cooperation_can_work" not in profile.learned_heuristics:
                profile.learned_heuristics.append("cooperation_can_work")
        
        elif memory.my_move == Move.DEFECT and memory.opponent_move == Move.COOPERATE:
            # I exploited them
            guilt_impact = (memory.my_payoff - memory.opponent_payoff) / 5.0
            guilt_memory = TraumaMemory(
                trauma_type="guilt",
                severity=guilt_impact * 0.5,  # Guilt is less severe than betrayal
                round_number=memory.round_number,
                emotional_impact=memory.emotional_impact * 0.5,
                description=f"Defected against cooperator, gained {memory.my_payoff - memory.opponent_payoff} extra points",
                timestamp=memory.timestamp
            )
            profile.trauma_memories.append(guilt_memory)
            
            profile.emotional_state = "conflicted"
            
            if "exploitation_feels_bad" not in profile.learned_heuristics:
                profile.learned_heuristics.append("exploitation_feels_bad")
        
        # Update personality traits based on accumulated experiences
        self._update_personality_traits()
    
    def _update_personality_traits(self) -> None:
        """Update personality traits based on psychological profile"""
        profile = self.psychological_profile
        
        # Clear old traits and rebuild
        profile.personality_traits.clear()
        
        if profile.loss_sensitivity > 2.5:
            profile.personality_traits.append("hypervigilant")
        if profile.trust_level < 0.2:
            profile.personality_traits.append("paranoid")
        if profile.trust_level > 0.8:
            profile.personality_traits.append("trusting")
        if len([t for t in profile.trauma_memories if t.trauma_type == "betrayal"]) > 3:
            profile.personality_traits.append("betrayal_sensitive")
        if profile.risk_tolerance < 0.3:
            profile.personality_traits.append("risk_averse")
    
    def observe_opponent_behavior(self, opponent_decision: Decision, opponent_reasoning: str) -> None:
        """Learn from observing opponent's behavior and reasoning"""
        
        observation_prompt = f"""
        {self._generate_personality_prompt()}
        
        You just observed your opponent's decision and reasoning:
        Decision: {opponent_decision.move.value}
        Confidence: {opponent_decision.confidence:.2f}
        Reasoning: {opponent_reasoning[:200]}...
        
        Based on this, what can you infer about their psychological state?
        Are they:
        - Overly cautious or loss averse?
        - Aggressive or risk-seeking?
        - Paranoid or trusting?
        - Rational or emotional?
        
        How should this influence your future interactions with them?
        Respond with key insights only.
        """
        
        response = self.llm_client.invoke(observation_prompt)
        
        # Store observation
        observation = {
            "round": len(self.memories),
            "opponent_move": opponent_decision.move.value,
            "opponent_confidence": opponent_decision.confidence,
            "psychological_insight": response.content,
            "timestamp": datetime.now()
        }
        
        self.observation_memory.append(observation)
        
        # Update learned heuristics based on observations
        insight_lower = response.content.lower()
        if "loss averse" in insight_lower and "opponents_often_loss_averse" not in self.psychological_profile.learned_heuristics:
            self.psychological_profile.learned_heuristics.append("opponents_often_loss_averse")
        if "paranoid" in insight_lower and "opponents_can_be_paranoid" not in self.psychological_profile.learned_heuristics:
            self.psychological_profile.learned_heuristics.append("opponents_can_be_paranoid")
        if "rational" in insight_lower and "some_opponents_are_rational" not in self.psychological_profile.learned_heuristics:
            self.psychological_profile.learned_heuristics.append("some_opponents_are_rational")
    
    def _get_recent_opponent_summary(self) -> str:
        """Get summary of recent opponent behavior"""
        recent_memories = self.get_recent_memories(5)
        if not recent_memories:
            return "No recent opponent behavior to analyze"
        
        opponent_moves = [m.opponent_move.value for m in recent_memories]
        cooperations = opponent_moves.count("cooperate")
        defections = opponent_moves.count("defect")
        
        summary = f"Last {len(recent_memories)} rounds: {cooperations} cooperations, {defections} defections. "
        summary += f"Pattern: {' -> '.join(opponent_moves[-3:])}"
        
        return summary
    
    def get_psychological_summary(self) -> Dict[str, Any]:
        """Get current psychological state summary"""
        profile = self.psychological_profile
        
        return {
            "agent_id": self.agent_id,
            "dominant_trait": profile.get_dominant_trait(),
            "trust_level": profile.trust_level,
            "loss_sensitivity": profile.loss_sensitivity,
            "risk_tolerance": profile.risk_tolerance,
            "emotional_state": profile.emotional_state,
            "personality_traits": profile.personality_traits,
            "learned_heuristics": profile.learned_heuristics,
            "trauma_count": len(profile.trauma_memories),
            "major_traumas": [
                t.description for t in sorted(profile.trauma_memories, key=lambda x: x.severity, reverse=True)[:3]
            ]
        }