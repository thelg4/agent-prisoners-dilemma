from typing import Dict, Any, List, Optional, Tuple
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
import logging
from datetime import datetime
from ..agents.emergent_bias_agent import EmergentBiasAgent
from ..population.population_dynamics import PopulationDynamics, PopulationSnapshot
from ..game.tournament import Tournament, TournamentResult
from ..game.game_state import GameState

class EmergentTournamentState(TypedDict):
    """State for emergent bias tournament workflow"""
    population_dynamics: PopulationDynamics
    current_generation: int
    max_generations: int
    interactions_per_generation: int
    rounds_per_interaction: int
    
    # Current interaction state
    current_interaction: int
    selected_agents: Optional[Tuple[EmergentBiasAgent, EmergentBiasAgent]]
    tournament_result: Optional[TournamentResult]
    
    # Results tracking
    generation_results: Annotated[List[Dict], add_messages]
    population_snapshots: Annotated[List[PopulationSnapshot], add_messages]
    
    # Workflow control
    workflow_status: str
    should_continue: bool

class EmergentTournamentWorkflow:
    """Advanced LangGraph workflow for emergent bias tournament system"""
    
    def __init__(self, population_size: int = 20, llm_client: Any = None):
        self.logger = logging.getLogger(__name__)
        self.population_size = population_size
        self.llm_client = llm_client
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the emergent bias tournament workflow"""
        
        workflow = StateGraph(EmergentTournamentState)
        
        # Add nodes
        workflow.add_node("initialize_population", self._initialize_population)
        workflow.add_node("take_snapshot", self._take_population_snapshot)
        workflow.add_node("select_agents", self._select_interaction_agents)
        workflow.add_node("run_interaction", self._run_agent_interaction)
        workflow.add_node("facilitate_observation", self._facilitate_mutual_observation)
        workflow.add_node("check_generation_complete", self._check_generation_complete)
        workflow.add_node("run_contagion", self._run_psychological_contagion)
        workflow.add_node("run_evolution", self._run_evolutionary_pressure)
        workflow.add_node("analyze_generation", self._analyze_generation_results)
        workflow.add_node("check_experiment_complete", self._check_experiment_complete)
        workflow.add_node("finalize_experiment", self._finalize_experiment)
        
        # Define workflow edges
        workflow.set_entry_point("initialize_population")
        workflow.add_edge("initialize_population", "take_snapshot")
        workflow.add_edge("take_snapshot", "select_agents")
        workflow.add_edge("select_agents", "run_interaction")
        workflow.add_edge("run_interaction", "facilitate_observation")
        workflow.add_edge("facilitate_observation", "check_generation_complete")
        
        # Conditional edges for generation management
        workflow.add_conditional_edges(
            "check_generation_complete",
            self._should_continue_generation,
            {
                "continue_interactions": "select_agents",
                "complete_generation": "run_contagion"
            }
        )
        
        workflow.add_edge("run_contagion", "run_evolution")
        workflow.add_edge("run_evolution", "analyze_generation")
        workflow.add_edge("analyze_generation", "check_experiment_complete")
        
        # Conditional edges for experiment management
        workflow.add_conditional_edges(
            "check_experiment_complete",
            self._should_continue_experiment,
            {
                "next_generation": "take_snapshot",
                "complete_experiment": "finalize_experiment"
            }
        )
        
        workflow.add_edge("finalize_experiment", END)
        
        return workflow.compile()
    
    def run_emergent_experiment(
        self,
        max_generations: int = 50,
        interactions_per_generation: int = 100,
        rounds_per_interaction: int = 100
    ) -> Dict[str, Any]:
        """Run complete emergent bias experiment"""
        
        initial_state = EmergentTournamentState(
            population_dynamics=PopulationDynamics(self.population_size, self.llm_client),
            current_generation=0,
            max_generations=max_generations,
            interactions_per_generation=interactions_per_generation,
            rounds_per_interaction=rounds_per_interaction,
            current_interaction=0,
            selected_agents=None,
            tournament_result=None,
            generation_results=[],
            population_snapshots=[],
            workflow_status="initializing",
            should_continue=True
        )
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        return result
    
    def _initialize_population(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Initialize the population with diverse agents"""
        
        self.logger.info(f"Initializing emergent bias experiment with {self.population_size} agents")
        
        # Population dynamics handles initialization
        state["workflow_status"] = "initialized"
        
        return state
    
    def _take_population_snapshot(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Take snapshot of current population psychological state"""
        
        population_dynamics = state["population_dynamics"]
        snapshot = population_dynamics._take_population_snapshot()
        
        state["population_snapshots"].append(snapshot)
        
        self.logger.info(f"Generation {state['current_generation']} snapshot taken")
        self.logger.info(f"  Dominant traits: {snapshot.dominant_traits}")
        self.logger.info(f"  Avg cooperation rate: {snapshot.avg_cooperation_rate:.3f}")
        
        return state
    
    def _select_interaction_agents(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Select two agents for interaction"""
        
        population = state["population_dynamics"].population
        
        # Select two random agents for interaction
        import random
        agent1, agent2 = random.sample(population, 2)
        
        state["selected_agents"] = (agent1, agent2)
        
        return state
    
    def _run_agent_interaction(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Run tournament between selected agents"""
        
        agent1, agent2 = state["selected_agents"]
        rounds = state["rounds_per_interaction"]
        generation = state["current_generation"]
        interaction = state["current_interaction"]
        
        # Create tournament
        tournament_id = f"gen_{generation:03d}_int_{interaction:03d}"
        tournament = Tournament(tournament_id)
        
        # Run tournament
        result = tournament.run_tournament(agent1, agent2, rounds)
        state["tournament_result"] = result
        
        return state
    
    def _facilitate_mutual_observation(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Let agents observe and learn from each other's behavior"""
        
        agent1, agent2 = state["selected_agents"]
        result = state["tournament_result"]
        
        # Use population dynamics method for observation
        state["population_dynamics"]._facilitate_mutual_observation(agent1, agent2, result)
        
        return state
    
    def _check_generation_complete(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Check if current generation has completed all interactions"""
        
        state["current_interaction"] += 1
        
        if state["current_interaction"] >= state["interactions_per_generation"]:
            state["current_interaction"] = 0  # Reset for next generation
            return state
        
        return state
    
    def _should_continue_generation(self, state: EmergentTournamentState) -> str:
        """Decide whether to continue interactions or complete generation"""
        
        if state["current_interaction"] == 0:  # Just reset, generation complete
            return "complete_generation"
        else:
            return "continue_interactions"
    
    def _run_psychological_contagion(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Run psychological contagion phase"""
        
        self.logger.info(f"Running psychological contagion for generation {state['current_generation']}")
        
        population_dynamics = state["population_dynamics"]
        population_dynamics._run_contagion_phase()
        
        return state
    
    def _run_evolutionary_pressure(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Apply evolutionary pressure every few generations"""
        
        generation = state["current_generation"]
        
        # Apply evolution every 10 generations
        if generation % 10 == 9:
            self.logger.info(f"Applying evolutionary pressure at generation {generation}")
            
            population_dynamics = state["population_dynamics"]
            population_dynamics._run_evolution_phase()
        
        return state
    
    def _analyze_generation_results(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Analyze results from completed generation"""
        
        generation = state["current_generation"]
        population_dynamics = state["population_dynamics"]
        
        # Calculate generation statistics
        population = population_dynamics.population
        
        generation_stats = {
            "generation": generation,
            "avg_score": sum(agent.total_score for agent in population) / len(population),
            "avg_cooperation_rate": sum(agent.get_cooperation_rate() for agent in population) / len(population),
            "psychological_diversity": len(set(
                agent.psychological_profile.get_dominant_trait() for agent in population
            )),
            "contagion_events_this_gen": len([
                e for e in population_dynamics.contagion_events 
                if e.generation == generation
            ])
        }
        
        state["generation_results"].append(generation_stats)
        
        self.logger.info(f"Generation {generation} analysis:")
        self.logger.info(f"  Avg score: {generation_stats['avg_score']:.2f}")
        self.logger.info(f"  Avg cooperation: {generation_stats['avg_cooperation_rate']:.3f}")
        self.logger.info(f"  Psychological diversity: {generation_stats['psychological_diversity']}")
        self.logger.info(f"  Contagion events: {generation_stats['contagion_events_this_gen']}")
        
        return state
    
    def _check_experiment_complete(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Check if experiment should continue"""
        
        state["current_generation"] += 1
        
        if state["current_generation"] >= state["max_generations"]:
            state["should_continue"] = False
        
        return state
    
    def _should_continue_experiment(self, state: EmergentTournamentState) -> str:
        """Decide whether to continue experiment or finish"""
        
        if state["should_continue"]:
            return "next_generation"
        else:
            return "complete_experiment"
    
    def _finalize_experiment(self, state: EmergentTournamentState) -> EmergentTournamentState:
        """Finalize experiment and prepare results"""
        
        self.logger.info("Finalizing emergent bias experiment")
        
        population_dynamics = state["population_dynamics"]
        
        # Generate final analysis
        evolution_analysis = population_dynamics.analyze_population_evolution()
        contagion_analysis = population_dynamics._analyze_contagion_patterns()
        
        # Store final results in state
        state["final_analysis"] = {
            "evolution_analysis": evolution_analysis,
            "contagion_analysis": contagion_analysis,
            "total_generations": state["current_generation"],
            "final_population_snapshot": state["population_snapshots"][-1] if state["population_snapshots"] else None
        }
        
        state["workflow_status"] = "completed"
        
        self.logger.info("Emergent bias experiment completed successfully")
        
        return state
    
    def create_experiment_summary(self, final_state: EmergentTournamentState) -> Dict[str, Any]:
        """Create comprehensive experiment summary"""
        
        if "final_analysis" not in final_state:
            return {"error": "Experiment not completed"}
        
        final_analysis = final_state["final_analysis"]
        evolution_analysis = final_analysis["evolution_analysis"]
        contagion_analysis = final_analysis["contagion_analysis"]
        
        # Extract key insights
        psychological_changes = evolution_analysis.get("psychological_changes", {})
        
        summary = {
            "experiment_overview": {
                "total_generations": final_analysis["total_generations"],
                "population_size": self.population_size,
                "total_interactions": final_analysis["total_generations"] * final_state["interactions_per_generation"]
            },
            "key_findings": {
                "emergent_traits": evolution_analysis.get("emerged_traits", []),
                "extinct_traits": evolution_analysis.get("extinct_traits", []),
                "final_dominant_traits": evolution_analysis.get("final_dominant_traits", []),
                "psychological_evolution": {
                    "trust_change": psychological_changes.get("trust_level_change", 0),
                    "loss_sensitivity_change": psychological_changes.get("loss_sensitivity_change", 0),
                    "cooperation_change": psychological_changes.get("cooperation_rate_change", 0)
                }
            },
            "contagion_insights": {
                "total_transmission_events": contagion_analysis.get("total_events", 0),
                "most_transmitted_trait": contagion_analysis.get("most_transmitted_trait"),
                "super_spreaders": contagion_analysis.get("super_spreaders", []),
                "avg_transmission_strength": contagion_analysis.get("avg_transmission_strength", 0)
            },
            "population_snapshots": final_state["population_snapshots"],
            "generation_results": final_state["generation_results"]
        }
        
        return summary