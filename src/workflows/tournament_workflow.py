from typing import Dict, Any, List, Optional, Tuple
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
import logging
from datetime import datetime
from ..agents.base_agent import BaseAgent, Memory
from ..game.tournament import Tournament, TournamentResult
from ..game.game_state import GameState
from .decision_workflow import DecisionWorkflow

class TournamentState(TypedDict):
    tournament_id: str
    agent1: BaseAgent
    agent2: BaseAgent
    num_rounds: int
    current_round: int
    tournament_result: Optional[TournamentResult]
    round_results: Annotated[List[Dict], add_messages]
    game_state: GameState
    workflow_status: str

class TournamentWorkflow:
    """LangGraph workflow for managing complete tournaments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_workflow = DecisionWorkflow()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the tournament management workflow"""
        
        workflow = StateGraph(TournamentState)
        
        # Add nodes
        workflow.add_node("initialize_tournament", self._initialize_tournament)
        workflow.add_node("play_round", self._play_round)
        workflow.add_node("update_memories", self._update_memories)
        workflow.add_node("check_completion", self._check_completion)
        workflow.add_node("finalize_tournament", self._finalize_tournament)
        
        # Define workflow edges
        workflow.set_entry_point("initialize_tournament")
        workflow.add_edge("initialize_tournament", "play_round")
        workflow.add_edge("play_round", "update_memories")
        workflow.add_edge("update_memories", "check_completion")
        
        # Conditional edge - continue or finish
        workflow.add_conditional_edges(
            "check_completion",
            self._should_continue_tournament,
            {
                "continue": "play_round",
                "finish": "finalize_tournament"
            }
        )
        
        workflow.add_edge("finalize_tournament", END)
        
        return workflow.compile()
    
    def run_tournament(
        self, 
        agent1: BaseAgent, 
        agent2: BaseAgent, 
        num_rounds: int = 1000,
        tournament_id: Optional[str] = None
    ) -> TournamentResult:
        """Run a complete tournament using LangGraph workflow"""
        
        if tournament_id is None:
            tournament_id = f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        initial_state = TournamentState(
            tournament_id=tournament_id,
            agent1=agent1,
            agent2=agent2,
            num_rounds=num_rounds,
            current_round=0,
            tournament_result=None,
            round_results=[],
            game_state=GameState(
                round_number=0,
                total_rounds=num_rounds,
                agent1_id=agent1.agent_id,
                agent2_id=agent2.agent_id,
                agent1_score=0.0,
                agent2_score=0.0,
                payoff_matrix={},
                round_history=[],
                agent1_memories=[],
                agent2_memories=[],
                tournament_id=tournament_id
            ),
            workflow_status="initializing"
        )
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        return result["tournament_result"]
    
    def _initialize_tournament(self, state: TournamentState) -> TournamentState:
        """Initialize the tournament"""
        self.logger.info(f"Initializing tournament {state['tournament_id']}")
        
        # Reset agent states
        state["agent1"].total_score = 0
        state["agent1"].round_count = 0
        state["agent1"].memories.clear()
        
        state["agent2"].total_score = 0
        state["agent2"].round_count = 0
        state["agent2"].memories.clear()
        
        state["workflow_status"] = "playing"
        
        return state
    
    def _play_round(self, state: TournamentState) -> TournamentState:
        """Play a single round"""
        state["current_round"] += 1
        round_num = state["current_round"]
        
        # Prepare game state for agents
        game_state_dict = {
            "round_number": round_num,
            "total_rounds": state["num_rounds"],
            "payoff_matrix": {
                ("cooperate", "cooperate"): (3, 3),
                ("cooperate", "defect"): (0, 5),
                ("defect", "cooperate"): (5, 0),
                ("defect", "defect"): (1, 1)
            }
        }
        
        # Get decisions from both agents using decision workflow
        decision1 = self.decision_workflow.run_decision_process(state["agent1"], game_state_dict)
        decision2 = self.decision_workflow.run_decision_process(state["agent2"], game_state_dict)
        
        # Calculate payoffs
        payoff_map = game_state_dict["payoff_matrix"]
        key = (decision1.move.value, decision2.move.value)
        payoffs = payoff_map[key]
        
        # Record round result
        round_result = {
            "round": round_num,
            "agent1_move": decision1.move.value,
            "agent2_move": decision2.move.value,
            "agent1_payoff": payoffs[0],
            "agent2_payoff": payoffs[1],
            "agent1_reasoning": decision1.reasoning_chain[-1].content if decision1.reasoning_chain else "",
            "agent2_reasoning": decision2.reasoning_chain[-1].content if decision2.reasoning_chain else ""
        }
        
        state["round_results"].append(round_result)
        
        # Update scores
        state["agent1"].total_score += payoffs[0]
        state["agent2"].total_score += payoffs[1]
        
        # Store decisions for memory update
        state["game_state"].current_decision_agent1 = decision1
        state["game_state"].current_decision_agent2 = decision2
        
        if round_num % 100 == 0:
            self.logger.info(f"Tournament {state['tournament_id']}: Round {round_num}/{state['num_rounds']}")
        
        return state
    
    def _update_memories(self, state: TournamentState) -> TournamentState:
        """Update agent memories with round results"""
        decision1 = state["game_state"].current_decision_agent1
        decision2 = state["game_state"].current_decision_agent2
        
        # Get the latest round result
        latest_round = state["round_results"][-1]
        
        # Create memories
        memory1 = Memory(
            round_number=state["current_round"],
            my_move=decision1.move,
            opponent_move=decision2.move,
            my_payoff=latest_round["agent1_payoff"],
            opponent_payoff=latest_round["agent2_payoff"],
            emotional_impact=self._calculate_emotional_impact(
                state["agent1"], latest_round["agent1_payoff"]
            ),
            timestamp=datetime.now()
        )
        
        memory2 = Memory(
            round_number=state["current_round"],
            my_move=decision2.move,
            opponent_move=decision1.move,
            my_payoff=latest_round["agent2_payoff"],
            opponent_payoff=latest_round["agent1_payoff"],
            emotional_impact=self._calculate_emotional_impact(
                state["agent2"], latest_round["agent2_payoff"]
            ),
            timestamp=datetime.now()
        )
        
        # Update agent memories
        state["agent1"].update_memory(memory1)
        state["agent2"].update_memory(memory2)
        
        return state
    
    def _check_completion(self, state: TournamentState) -> TournamentState:
        """Check if tournament is complete"""
        if state["current_round"] >= state["num_rounds"]:
            state["workflow_status"] = "completing"
        
        return state
    
    def _should_continue_tournament(self, state: TournamentState) -> str:
        """Decide whether to continue or finish the tournament"""
        if state["current_round"] >= state["num_rounds"]:
            return "finish"
        else:
            return "continue"
    
    def _finalize_tournament(self, state: TournamentState) -> TournamentState:
        """Finalize tournament and create results"""
        self.logger.info(f"Finalizing tournament {state['tournament_id']}")
        
        # Calculate cooperation rates
        agent1_cooperations = sum(1 for r in state["round_results"] if r["agent1_move"] == "cooperate")
        agent2_cooperations = sum(1 for r in state["round_results"] if r["agent2_move"] == "cooperate")
        
        agent1_coop_rate = agent1_cooperations / len(state["round_results"])
        agent2_coop_rate = agent2_cooperations / len(state["round_results"])
        
        # Create tournament result
        tournament_result = TournamentResult(
            tournament_id=state["tournament_id"],
            agent1_id=state["agent1"].agent_id,
            agent2_id=state["agent2"].agent_id,
            agent1_type=state["agent1"].agent_type,
            agent2_type=state["agent2"].agent_type,
            total_rounds=state["num_rounds"],
            agent1_final_score=state["agent1"].total_score,
            agent2_final_score=state["agent2"].total_score,
            agent1_cooperation_rate=agent1_coop_rate,
            agent2_cooperation_rate=agent2_coop_rate,
            cooperation_by_round=[(r["agent1_move"] == "cooperate", r["agent2_move"] == "cooperate") 
                                 for r in state["round_results"]],
            final_payoffs=(state["agent1"].total_score, state["agent2"].total_score),
            tournament_duration=0.0,  # Would need to track actual time
            timestamp=datetime.now()
        )
        
        state["tournament_result"] = tournament_result
        state["workflow_status"] = "completed"
        
        self.logger.info(f"Tournament completed: {state['agent1'].agent_id}={state['agent1'].total_score}, "
                        f"{state['agent2'].agent_id}={state['agent2'].total_score}")
        
        return state
    
    def _calculate_emotional_impact(self, agent: BaseAgent, payoff: float) -> float:
        """Calculate emotional impact for memory storage"""
        if hasattr(agent, 'reference_point'):
            reference = agent.reference_point
        else:
            reference = 2.0  # Neutral expectation
        
        impact = payoff - reference
        
        # Amplify negative emotions for loss-averse agents
        if hasattr(agent, 'loss_aversion') and impact < 0:
            return impact * agent.loss_aversion.loss_coefficient
        
        return impact