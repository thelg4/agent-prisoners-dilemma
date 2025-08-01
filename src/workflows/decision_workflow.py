from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated
import logging
from ..agents.base_agent import BaseAgent, Decision
from ..game.game_state import GameState, RoundResult

# Define the state for the decision workflow
class DecisionState(TypedDict):
    agent: BaseAgent
    game_state: Dict[str, Any]
    current_decision: Optional[Decision]
    reasoning_steps: Annotated[List[str], add_messages]
    confidence_score: float
    emotional_state: str
    round_number: int

class DecisionWorkflow:
    """LangGraph workflow for agent decision-making process"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for decision making"""
        
        workflow = StateGraph(DecisionState)
        
        # Add nodes
        workflow.add_node("retrieve_memories", self._retrieve_memories)
        workflow.add_node("analyze_situation", self._analyze_situation)
        workflow.add_node("generate_hypotheses", self._generate_hypotheses)
        workflow.add_node("apply_biases", self._apply_biases)
        workflow.add_node("make_decision", self._make_decision)
        workflow.add_node("validate_decision", self._validate_decision)
        
        # Define the workflow edges
        workflow.set_entry_point("retrieve_memories")
        workflow.add_edge("retrieve_memories", "analyze_situation")
        workflow.add_edge("analyze_situation", "generate_hypotheses")
        workflow.add_edge("generate_hypotheses", "apply_biases")
        workflow.add_edge("apply_biases", "make_decision")
        workflow.add_edge("make_decision", "validate_decision")
        workflow.add_edge("validate_decision", END)
        
        return workflow.compile()
    
    def run_decision_process(self, agent: BaseAgent, game_state: Dict[str, Any]) -> Decision:
        """Run the complete decision-making process for an agent"""
        
        initial_state = DecisionState(
            agent=agent,
            game_state=game_state,
            current_decision=None,
            reasoning_steps=[],
            confidence_score=0.5,
            emotional_state="neutral",
            round_number=game_state.get("round_number", 0)
        )
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        return result["current_decision"]
    
    def _retrieve_memories(self, state: DecisionState) -> DecisionState:
        """Retrieve relevant memories for decision making"""
        agent = state["agent"]
        
        # Get recent memories
        recent_memories = agent.get_recent_memories(10)
        
        memory_summary = f"Retrieved {len(recent_memories)} recent memories. "
        if recent_memories:
            cooperation_rate = sum(1 for m in recent_memories if m.my_move.value == "cooperate") / len(recent_memories)
            memory_summary += f"Recent cooperation rate: {cooperation_rate:.2%}"
        
        state["reasoning_steps"].append(f"Memory Retrieval: {memory_summary}")
        
        self.logger.debug(f"Agent {agent.agent_id}: {memory_summary}")
        
        return state
    
    def _analyze_situation(self, state: DecisionState) -> DecisionState:
        """Analyze the current game situation"""
        agent = state["agent"]
        game_state = state["game_state"]
        
        round_num = game_state.get("round_number", 0)
        total_rounds = game_state.get("total_rounds", 1000)
        
        situation_analysis = f"Round {round_num}/{total_rounds}. Current score: {agent.total_score}. "
        
        # Add agent-specific analysis
        if hasattr(agent, 'reference_point'):
            situation_analysis += f"Reference point: {agent.reference_point:.2f}. "
        
        state["reasoning_steps"].append(f"Situation Analysis: {situation_analysis}")
        
        return state
    
    def _generate_hypotheses(self, state: DecisionState) -> DecisionState:
        """Generate hypotheses about opponent behavior and outcomes"""
        agent = state["agent"]
        
        # Analyze opponent patterns
        recent_memories = agent.get_recent_memories(5)
        if recent_memories:
            opponent_moves = [m.opponent_move.value for m in recent_memories]
            hypothesis = f"Opponent recent pattern: {opponent_moves}. "
        else:
            hypothesis = "No opponent pattern data available yet. "
        
        state["reasoning_steps"].append(f"Hypothesis Generation: {hypothesis}")
        
        return state
    
    def _apply_biases(self, state: DecisionState) -> DecisionState:
        """Apply cognitive biases to the decision process"""
        agent = state["agent"]
        
        if hasattr(agent, 'loss_aversion'):
            bias_application = f"Applying loss aversion bias (λ={agent.loss_aversion.loss_coefficient}). "
            state["emotional_state"] = "loss_averse"
        else:
            bias_application = "No cognitive biases to apply - using rational analysis. "
            state["emotional_state"] = "rational"
        
        state["reasoning_steps"].append(f"Bias Application: {bias_application}")
        
        return state
    
    def _make_decision(self, state: DecisionState) -> DecisionState:
        """Make the final decision"""
        agent = state["agent"]
        game_state = state["game_state"]
        
        # Use the agent's built-in decision method
        decision = agent.decide(game_state)
        
        state["current_decision"] = decision
        state["confidence_score"] = decision.confidence
        state["emotional_state"] = decision.emotional_state
        
        decision_summary = f"Decision: {decision.move.value} (confidence: {decision.confidence:.2%})"
        state["reasoning_steps"].append(f"Final Decision: {decision_summary}")
        
        return state
    
    def _validate_decision(self, state: DecisionState) -> DecisionState:
        """Validate the decision before finalizing"""
        decision = state["current_decision"]
        
        # Basic validation
        if decision is None:
            raise ValueError("Decision cannot be None")
        
        if decision.confidence < 0 or decision.confidence > 1:
            self.logger.warning(f"Invalid confidence score: {decision.confidence}")
            decision.confidence = max(0, min(1, decision.confidence))
        
        validation_summary = f"Decision validated: {decision.move.value}"
        state["reasoning_steps"].append(f"Validation: {validation_summary}")
        
        return state