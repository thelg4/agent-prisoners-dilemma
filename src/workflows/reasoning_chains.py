from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from ..agents.base_agent import ReasoningStep, BaseAgent

@dataclass
class ReasoningChain:
    """A complete chain of reasoning steps"""
    agent_id: str
    round_number: int
    steps: List[ReasoningStep]
    final_decision: str
    confidence: float
    total_reasoning_time: float
    
    def get_summary(self) -> str:
        """Get a summary of the reasoning chain"""
        if not self.steps:
            return "No reasoning steps recorded"
        
        summary_parts = []
        for step in self.steps:
            summary_parts.append(f"{step.step_type}: {step.content[:100]}...")
        
        return " → ".join(summary_parts)
    
    def get_step_by_type(self, step_type: str) -> Optional[ReasoningStep]:
        """Get the first step of a specific type"""
        for step in self.steps:
            if step.step_type == step_type:
                return step
        return None

class ReasoningChains:
    """Utility class for managing and analyzing reasoning chains"""
    
    def __init__(self):
        self.chains: Dict[str, List[ReasoningChain]] = {}
    
    def add_chain(self, chain: ReasoningChain) -> None:
        """Add a reasoning chain"""
        if chain.agent_id not in self.chains:
            self.chains[chain.agent_id] = []
        self.chains[chain.agent_id].append(chain)
    
    def get_chains_for_agent(self, agent_id: str) -> List[ReasoningChain]:
        """Get all reasoning chains for a specific agent"""
        return self.chains.get(agent_id, [])
    
    def analyze_reasoning_patterns(self, agent_id: str) -> Dict[str, Any]:
        """Analyze reasoning patterns for an agent"""
        chains = self.get_chains_for_agent(agent_id)
        
        if not chains:
            return {"error": "No reasoning chains found for agent"}
        
        # Calculate statistics
        avg_steps = sum(len(chain.steps) for chain in chains) / len(chains)
        avg_confidence = sum(chain.confidence for chain in chains) / len(chains)
        avg_reasoning_time = sum(chain.total_reasoning_time for chain in chains) / len(chains)
        
        # Analyze step types
        step_type_counts = {}
        for chain in chains:
            for step in chain.steps:
                step_type_counts[step.step_type] = step_type_counts.get(step.step_type, 0) + 1
        
        # Analyze decision patterns
        decisions = [chain.final_decision for chain in chains]
        cooperate_rate = decisions.count("cooperate") / len(decisions) if decisions else 0
        
        return {
            "total_chains": len(chains),
            "avg_steps_per_chain": avg_steps,
            "avg_confidence": avg_confidence,
            "avg_reasoning_time": avg_reasoning_time,
            "step_type_distribution": step_type_counts,
            "cooperation_rate": cooperate_rate,
            "most_common_decision": max(set(decisions), key=decisions.count) if decisions else None
        }
    
    def compare_reasoning_complexity(self, agent1_id: str, agent2_id: str) -> Dict[str, Any]:
        """Compare reasoning complexity between two agents"""
        analysis1 = self.analyze_reasoning_patterns(agent1_id)
        analysis2 = self.analyze_reasoning_patterns(agent2_id)
        
        if "error" in analysis1 or "error" in analysis2:
            return {"error": "Insufficient data for comparison"}
        
        return {
            "agent1_complexity": analysis1["avg_steps_per_chain"],
            "agent2_complexity": analysis2["avg_steps_per_chain"],
            "complexity_difference": analysis1["avg_steps_per_chain"] - analysis2["avg_steps_per_chain"],
            "agent1_confidence": analysis1["avg_confidence"],
            "agent2_confidence": analysis2["avg_confidence"],
            "confidence_difference": analysis1["avg_confidence"] - analysis2["avg_confidence"]
        }