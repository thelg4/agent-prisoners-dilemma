from langgraph.graph import StateGraph, END
from .models import GameState, Move, GameResult, AgentType
from .agents import create_agent, BaseAgent
from .payoffs import PayoffMatrix
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PrisonersDilemmaGraph:
    """LangGraph workflow for the Prisoner's Dilemma game"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(GameState)
        
        # Add nodes
        workflow.add_node("initialize_agents", self.initialize_agents)
        workflow.add_node("get_agent_moves", self.get_agent_moves)
        workflow.add_node("calculate_payoffs", self.calculate_payoffs)
        workflow.add_node("update_agent_states", self.update_agent_states)
        workflow.add_node("check_game_end", self.check_game_end)
        workflow.add_node("finalize_game", self.finalize_game)
        
        # Add edges
        workflow.add_edge("initialize_agents", "get_agent_moves")
        workflow.add_edge("get_agent_moves", "calculate_payoffs")
        workflow.add_edge("calculate_payoffs", "update_agent_states")
        workflow.add_edge("update_agent_states", "check_game_end")
        
        # Conditional edge from check_game_end
        workflow.add_conditional_edges(
            "check_game_end",
            self.should_continue,
            {
                "continue": "get_agent_moves",
                "end": "finalize_game"
            }
        )
        
        workflow.add_edge("finalize_game", END)
        workflow.set_entry_point("initialize_agents")
        
        return workflow.compile()
    
    def initialize_agents(self, state: GameState) -> GameState:
        """Initialize the agents based on their types"""
        logger.info(f"Initializing agents for game {state['game_id']}")
        
        agent1_state = state["agent1"]
        agent2_state = state["agent2"]
        
        # Create agent instances
        self.agents[agent1_state.agent_id] = create_agent(
            agent1_state.agent_id, 
            agent1_state.agent_type, 
            agent1_state.name
        )
        
        self.agents[agent2_state.agent_id] = create_agent(
            agent2_state.agent_id, 
            agent2_state.agent_type, 
            agent2_state.name
        )
        
        logger.info(f"Created {agent1_state.agent_type.value} agent: {agent1_state.name}")
        logger.info(f"Created {agent2_state.agent_type.value} agent: {agent2_state.name}")
        
        state["updated_at"] = datetime.now()
        return state
    
    def get_agent_moves(self, state: GameState) -> GameState:
        """Get moves from both agents simultaneously"""
        logger.debug(f"Getting moves for round {state['current_round']}")
        
        agent1_state = state["agent1"]
        agent2_state = state["agent2"]
        
        try:
            # Get moves from agents
            agent1_move = self.agents[agent1_state.agent_id].decide_move(state, agent1_state)
            agent2_move = self.agents[agent2_state.agent_id].decide_move(state, agent2_state)
            
            logger.debug(f"Agent 1 ({agent1_state.name}): {agent1_move.value}")
            logger.debug(f"Agent 2 ({agent2_state.name}): {agent2_move.value}")
            
            # Store moves
            state["current_moves"] = {
                agent1_state.agent_id: agent1_move,
                agent2_state.agent_id: agent2_move
            }
            
        except Exception as e:
            logger.error(f"Error getting agent moves: {e}")
            # Fallback to random moves if agents fail
            agent1_move = Move.COOPERATE
            agent2_move = Move.COOPERATE
            state["current_moves"] = {
                agent1_state.agent_id: agent1_move,
                agent2_state.agent_id: agent2_move
            }
        
        state["updated_at"] = datetime.now()
        return state
    
    def calculate_payoffs(self, state: GameState) -> GameState:
        """Calculate payoffs for the current round"""
        logger.debug(f"Calculating payoffs for round {state['current_round']}")
        
        agent1_state = state["agent1"]
        agent2_state = state["agent2"]
        current_moves = state["current_moves"]
        
        agent1_move = current_moves[agent1_state.agent_id]
        agent2_move = current_moves[agent2_state.agent_id]
        
        # Get raw payoffs from the matrix
        raw_payoffs = PayoffMatrix.get_payoffs(agent1_move, agent2_move)
        agent1_raw_payoff = raw_payoffs[0]
        agent2_raw_payoff = raw_payoffs[1]
        
        # Apply loss aversion if the agent is human
        loss_factor = state["loss_aversion_factor"]
        reference_point = state["reference_point"]
        
        # Agent 1 adjusted payoff
        if agent1_state.agent_type == AgentType.HUMAN:
            agent1_adjusted = PayoffMatrix.apply_loss_aversion(
                agent1_raw_payoff, loss_factor, reference_point
            )
        else:
            agent1_adjusted = agent1_raw_payoff
        
        # Agent 2 adjusted payoff  
        if agent2_state.agent_type == AgentType.HUMAN:
            agent2_adjusted = PayoffMatrix.apply_loss_aversion(
                agent2_raw_payoff, loss_factor, reference_point
            )
        else:
            agent2_adjusted = agent2_raw_payoff
        
        # Create game result
        result = GameResult(
            agent1_move=agent1_move,
            agent2_move=agent2_move,
            agent1_payoff=agent1_raw_payoff,
            agent2_payoff=agent2_raw_payoff,
            agent1_adjusted_payoff=agent1_adjusted,
            agent2_adjusted_payoff=agent2_adjusted,
            round_number=state["current_round"],
            timestamp=datetime.now()
        )
        
        state["game_history"].append(result)
        
        logger.debug(f"Round {state['current_round']} result: "
                    f"{agent1_move.value} vs {agent2_move.value}, "
                    f"Payoffs: ({agent1_raw_payoff}, {agent2_raw_payoff}), "
                    f"Adjusted: ({agent1_adjusted:.2f}, {agent2_adjusted:.2f})")
        
        state["updated_at"] = datetime.now()
        return state
    
    def update_agent_states(self, state: GameState) -> GameState:
        """Update agent states with the results of the current round"""
        logger.debug(f"Updating agent states after round {state['current_round']}")
        
        latest_result = state["game_history"][-1]
        agent1_state = state["agent1"]
        agent2_state = state["agent2"]
        
        # Update agent 1
        agent1_state.moves_history.append(latest_result.agent1_move)
        agent1_state.opponent_moves_history.append(latest_result.agent2_move)
        agent1_state.total_payoff += latest_result.agent1_payoff
        agent1_state.adjusted_total_payoff += latest_result.agent1_adjusted_payoff
        
        # Update agent 2
        agent2_state.moves_history.append(latest_result.agent2_move)
        agent2_state.opponent_moves_history.append(latest_result.agent1_move)
        agent2_state.total_payoff += latest_result.agent2_payoff
        agent2_state.adjusted_total_payoff += latest_result.agent2_adjusted_payoff
        
        # Update statistics
        self._update_agent_statistics(agent1_state)
        self._update_agent_statistics(agent2_state)
        
        # Increment round counter
        state["current_round"] += 1
        state["updated_at"] = datetime.now()
        
        return state
    
    def _update_agent_statistics(self, agent_state):
        """Update cooperation/defection rates and averages"""
        total_moves = len(agent_state.moves_history)
        if total_moves > 0:
            cooperations = agent_state.moves_history.count(Move.COOPERATE)
            agent_state.cooperation_rate = cooperations / total_moves
            agent_state.defection_rate = 1 - agent_state.cooperation_rate
            agent_state.average_payoff_per_round = agent_state.total_payoff / total_moves
    
    def check_game_end(self, state: GameState) -> GameState:
        """Check if the game should end"""
        if state["current_round"] > state["max_rounds"]:
            state["game_complete"] = True
            
            # Determine winner based on adjusted payoffs (accounting for loss aversion)
            agent1_score = state["agent1"].adjusted_total_payoff
            agent2_score = state["agent2"].adjusted_total_payoff
            
            if agent1_score > agent2_score:
                state["winner"] = state["agent1"].agent_id
            elif agent2_score > agent1_score:
                state["winner"] = state["agent2"].agent_id
            else:
                state["winner"] = "tie"
            
            logger.info(f"Game {state['game_id']} completed. Winner: {state['winner']}")
        
        state["updated_at"] = datetime.now()
        return state
    
    def should_continue(self, state: GameState) -> str:
        """Decide whether to continue or end the game"""
        return "end" if state["game_complete"] else "continue"
    
    def finalize_game(self, state: GameState) -> GameState:
        """Generate final game summary and statistics"""
        logger.info(f"Finalizing game {state['game_id']}")
        
        agent1_state = state["agent1"]
        agent2_state = state["agent2"]
        game_history = state["game_history"]
        
        # Calculate additional statistics
        mutual_cooperation_rounds = sum(
            1 for result in game_history
            if result.agent1_move == Move.COOPERATE and result.agent2_move == Move.COOPERATE
        )
        
        mutual_defection_rounds = sum(
            1 for result in game_history
            if result.agent1_move == Move.DEFECT and result.agent2_move == Move.DEFECT
        )
        
        agent1_exploited_rounds = sum(
            1 for result in game_history
            if result.agent1_move == Move.COOPERATE and result.agent2_move == Move.DEFECT
        )
        
        agent2_exploited_rounds = sum(
            1 for result in game_history
            if result.agent1_move == Move.DEFECT and result.agent2_move == Move.COOPERATE
        )
        
        # Calculate loss aversion impact
        total_loss_aversion_impact_agent1 = sum(
            result.agent1_adjusted_payoff - result.agent1_payoff for result in game_history
        )
        
        total_loss_aversion_impact_agent2 = sum(
            result.agent2_adjusted_payoff - result.agent2_payoff for result in game_history
        )
        
        # Generate comprehensive summary
        state["game_summary"] = {
            "game_info": {
                "total_rounds": len(game_history),
                "max_rounds": state["max_rounds"],
                "loss_aversion_factor": state["loss_aversion_factor"],
                "reference_point": state["reference_point"],
                "duration_seconds": (state["updated_at"] - state["created_at"]).total_seconds()
            },
            "winner": state["winner"],
            "agents": {
                "agent1": {
                    "id": agent1_state.agent_id,
                    "name": agent1_state.name,
                    "type": agent1_state.agent_type.value,
                    "total_payoff": agent1_state.total_payoff,
                    "adjusted_payoff": agent1_state.adjusted_total_payoff,
                    "cooperation_rate": agent1_state.cooperation_rate,
                    "defection_rate": agent1_state.defection_rate,
                    "average_payoff": agent1_state.average_payoff_per_round,
                    "loss_aversion_impact": total_loss_aversion_impact_agent1
                },
                "agent2": {
                    "id": agent2_state.agent_id,
                    "name": agent2_state.name,
                    "type": agent2_state.agent_type.value,
                    "total_payoff": agent2_state.total_payoff,
                    "adjusted_payoff": agent2_state.adjusted_total_payoff,
                    "cooperation_rate": agent2_state.cooperation_rate,
                    "defection_rate": agent2_state.defection_rate,
                    "average_payoff": agent2_state.average_payoff_per_round,
                    "loss_aversion_impact": total_loss_aversion_impact_agent2
                }
            },
            "round_types": {
                "mutual_cooperation": mutual_cooperation_rounds,
                "mutual_defection": mutual_defection_rounds,
                "agent1_exploited": agent1_exploited_rounds,
                "agent2_exploited": agent2_exploited_rounds
            },
            "behavioral_insights": self._generate_behavioral_insights(state)
        }
        
        state["updated_at"] = datetime.now()
        return state
    
    def _generate_behavioral_insights(self, state: GameState) -> Dict[str, Any]:
        """Generate insights about agent behavior patterns"""
        agent1_state = state["agent1"]
        agent2_state = state["agent2"]
        game_history = state["game_history"]
        
        insights = {}
        
        # Cooperation evolution over time
        if len(game_history) >= 5:
            early_rounds = game_history[:3]
            late_rounds = game_history[-3:]
            
            agent1_early_coop = sum(1 for r in early_rounds if r.agent1_move == Move.COOPERATE) / len(early_rounds)
            agent1_late_coop = sum(1 for r in late_rounds if r.agent1_move == Move.COOPERATE) / len(late_rounds)
            
            agent2_early_coop = sum(1 for r in early_rounds if r.agent2_move == Move.COOPERATE) / len(early_rounds)
            agent2_late_coop = sum(1 for r in late_rounds if r.agent2_move == Move.COOPERATE) / len(late_rounds)
            
            insights["cooperation_evolution"] = {
                "agent1": {
                    "early_cooperation_rate": agent1_early_coop,
                    "late_cooperation_rate": agent1_late_coop,
                    "cooperation_change": agent1_late_coop - agent1_early_coop
                },
                "agent2": {
                    "early_cooperation_rate": agent2_early_coop,
                    "late_cooperation_rate": agent2_late_coop,
                    "cooperation_change": agent2_late_coop - agent2_early_coop
                }
            }
        
        # Reciprocity analysis
        reciprocity_scores = self._calculate_reciprocity(game_history)
        insights["reciprocity"] = reciprocity_scores
        
        # Loss aversion effectiveness
        if agent1_state.agent_type == AgentType.HUMAN or agent2_state.agent_type == AgentType.HUMAN:
            insights["loss_aversion_analysis"] = self._analyze_loss_aversion_impact(state)
        
        return insights
    
    def _calculate_reciprocity(self, game_history: list) -> Dict[str, float]:
        """Calculate how much each agent reciprocates the other's moves"""
        if len(game_history) < 2:
            return {"agent1_reciprocity": 0.0, "agent2_reciprocity": 0.0}
        
        agent1_reciprocity = 0
        agent2_reciprocity = 0
        valid_rounds = 0
        
        for i in range(1, len(game_history)):
            prev_round = game_history[i-1]
            curr_round = game_history[i]
            
            # Agent 1 reciprocating Agent 2's previous move
            if prev_round.agent2_move == curr_round.agent1_move:
                agent1_reciprocity += 1
            
            # Agent 2 reciprocating Agent 1's previous move
            if prev_round.agent1_move == curr_round.agent2_move:
                agent2_reciprocity += 1
            
            valid_rounds += 1
        
        return {
            "agent1_reciprocity": agent1_reciprocity / valid_rounds if valid_rounds > 0 else 0.0,
            "agent2_reciprocity": agent2_reciprocity / valid_rounds if valid_rounds > 0 else 0.0
        }
    
    def _analyze_loss_aversion_impact(self, state: GameState) -> Dict[str, Any]:
        """Analyze the impact of loss aversion on the game"""
        game_history = state["game_history"]
        agent1_state = state["agent1"]
        agent2_state = state["agent2"]
        
        analysis = {}
        
        # Count how many times each agent experienced losses
        agent1_losses = sum(1 for r in game_history if r.agent1_payoff < state["reference_point"])
        agent2_losses = sum(1 for r in game_history if r.agent2_payoff < state["reference_point"])
        
        # Calculate total loss aversion penalty
        agent1_penalty = sum(
            r.agent1_payoff - r.agent1_adjusted_payoff for r in game_history
            if r.agent1_adjusted_payoff < r.agent1_payoff
        )
        
        agent2_penalty = sum(
            r.agent2_payoff - r.agent2_adjusted_payoff for r in game_history
            if r.agent2_adjusted_payoff < r.agent2_payoff
        )
        
        analysis = {
            "loss_frequency": {
                "agent1": agent1_losses,
                "agent2": agent2_losses
            },
            "total_loss_penalty": {
                "agent1": agent1_penalty,
                "agent2": agent2_penalty
            },
            "average_loss_penalty_per_round": {
                "agent1": agent1_penalty / len(game_history) if game_history else 0,
                "agent2": agent2_penalty / len(game_history) if game_history else 0
            }
        }
        
        # Determine if loss aversion affected the outcome
        raw_winner = None
        if agent1_state.total_payoff > agent2_state.total_payoff:
            raw_winner = agent1_state.agent_id
        elif agent2_state.total_payoff > agent1_state.total_payoff:
            raw_winner = agent2_state.agent_id
        else:
            raw_winner = "tie"
        
        analysis["outcome_impact"] = {
            "raw_payoff_winner": raw_winner,
            "adjusted_payoff_winner": state["winner"],
            "winner_changed_due_to_loss_aversion": raw_winner != state["winner"]
        }
        
        return analysis