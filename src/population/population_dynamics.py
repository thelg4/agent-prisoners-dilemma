from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import logging
from ..agents.emergent_bias_agent import EmergentBiasAgent
from ..game.tournament import Tournament, TournamentResult
from ..analysis.behavioral_metrics import BehavioralMetrics

@dataclass
class PopulationSnapshot:
    """Snapshot of population psychological state at a point in time"""
    generation: int
    population_size: int
    psychological_distribution: Dict[str, int]  # trait -> count
    avg_trust_level: float
    avg_loss_sensitivity: float
    avg_cooperation_rate: float
    dominant_traits: List[str]
    timestamp: datetime

@dataclass
class BiasContagionEvent:
    """Records when psychological traits spread between agents"""
    source_agent: str
    target_agent: str
    transmitted_trait: str
    generation: int
    interaction_round: int
    transmission_strength: float
    timestamp: datetime

class PopulationDynamics:
    """Manages population-level emergent bias dynamics"""
    
    def __init__(self, population_size: int = 20, llm_client: Any = None):
        self.population_size = population_size
        self.llm_client = llm_client
        self.population: List[EmergentBiasAgent] = []
        self.generation = 0
        self.population_history: List[PopulationSnapshot] = []
        self.contagion_events: List[BiasContagionEvent] = []
        self.behavioral_metrics = BehavioralMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Initialize population with diverse starting personalities"""
        starting_personalities = [
            "optimistic", "cautious", "analytical", "intuitive", 
            "trusting", "skeptical", "balanced", "curious"
        ]
        
        self.population = []
        for i in range(self.population_size):
            personality = starting_personalities[i % len(starting_personalities)]
            agent = EmergentBiasAgent(
                agent_id=f"agent_{i:03d}",
                llm_client=self.llm_client,
                initial_personality=personality
            )
            self.population.append(agent)
        
        self.logger.info(f"Initialized population of {self.population_size} agents")
    
    def run_evolutionary_simulation(
        self, 
        generations: int = 50,
        interactions_per_generation: int = 100,
        rounds_per_interaction: int = 100
    ) -> List[PopulationSnapshot]:
        """Run complete evolutionary simulation"""
        
        self.logger.info(f"Starting evolutionary simulation: {generations} generations")
        
        for gen in range(generations):
            self.generation = gen
            
            # Take population snapshot
            snapshot = self._take_population_snapshot()
            self.population_history.append(snapshot)
            
            # Run interaction phase
            self._run_interaction_phase(interactions_per_generation, rounds_per_interaction)
            
            # Psychological influence and contagion phase
            self._run_contagion_phase()
            
            # Evolution phase (successful traits spread)
            if gen % 10 == 9:  # Every 10 generations
                self._run_evolution_phase()
            
            # Log progress
            if gen % 10 == 0:
                self._log_generation_summary(snapshot)
        
        # Final snapshot
        final_snapshot = self._take_population_snapshot()
        self.population_history.append(final_snapshot)
        
        self.logger.info("Evolutionary simulation completed")
        return self.population_history
    
    def _run_interaction_phase(self, num_interactions: int, rounds_per_interaction: int) -> None:
        """Run random interactions between agents"""
        
        interaction_results = []
        
        for interaction in range(num_interactions):
            # Select two random agents
            agent1, agent2 = random.sample(self.population, 2)
            
            # Run tournament between them
            tournament = Tournament(f"gen_{self.generation}_int_{interaction}")
            result = tournament.run_tournament(agent1, agent2, rounds_per_interaction)
            interaction_results.append(result)
            
            # Agents observe each other's behavior
            self._facilitate_mutual_observation(agent1, agent2, result)
        
        self.logger.debug(f"Generation {self.generation}: Completed {num_interactions} interactions")
    
    def _facilitate_mutual_observation(
        self, 
        agent1: EmergentBiasAgent, 
        agent2: EmergentBiasAgent, 
        result: TournamentResult
    ) -> None:
        """Let agents observe and learn from each other's behavior"""
        
        # Get sample decisions and reasoning from each agent
        if result.round_details:
            # Sample some rounds for observation
            sample_indices = random.sample(range(len(result.round_details)), min(5, len(result.round_details)))
            
            for idx in sample_indices:
                round_detail = result.round_details[idx]
                
                # Agent1 observes Agent2
                mock_decision2 = type('Decision', (), {
                    'move': type('Move', (), {'value': round_detail.agent2_move})(),
                    'confidence': round_detail.agent2_confidence
                })()
                agent1.observe_opponent_behavior(mock_decision2, round_detail.agent2_reasoning)
                
                # Agent2 observes Agent1
                mock_decision1 = type('Decision', (), {
                    'move': type('Move', (), {'value': round_detail.agent1_move})(),
                    'confidence': round_detail.agent1_confidence
                })()
                agent2.observe_opponent_behavior(mock_decision1, round_detail.agent1_reasoning)
    
    def _run_contagion_phase(self) -> None:
        """Run psychological contagion phase where traits spread"""
        
        # Identify successful agents (top performers in recent interactions)
        successful_agents = self._identify_successful_agents()
        
        # Each agent has a chance to be influenced by successful agents
        for agent in self.population:
            if agent not in successful_agents:
                # Select a random successful agent to potentially learn from
                if successful_agents:
                    role_model = random.choice(successful_agents)
                    self._attempt_psychological_contagion(agent, role_model)
    
    def _identify_successful_agents(self) -> List[EmergentBiasAgent]:
        """Identify agents who have been most successful recently"""
        
        # Sort agents by total score (simple metric for now)
        sorted_agents = sorted(self.population, key=lambda a: a.total_score, reverse=True)
        
        # Return top 25%
        top_count = max(1, self.population_size // 4)
        return sorted_agents[:top_count]
    
    def _attempt_psychological_contagion(
        self, 
        target_agent: EmergentBiasAgent, 
        source_agent: EmergentBiasAgent
    ) -> None:
        """Attempt to transmit psychological traits between agents"""
        
        source_profile = source_agent.psychological_profile
        target_profile = target_agent.psychological_profile
        
        # Calculate contagion probability based on success difference
        success_diff = source_agent.total_score - target_agent.total_score
        base_probability = 0.1 + min(0.4, success_diff / 1000.0)  # Max 50% chance
        
        if random.random() < base_probability:
            # Determine what trait to transmit
            trait_to_transmit = self._select_trait_for_transmission(source_profile, target_profile)
            
            if trait_to_transmit:
                # Apply trait transmission
                transmission_strength = random.uniform(0.1, 0.3)
                self._apply_trait_transmission(target_profile, source_profile, trait_to_transmit, transmission_strength)
                
                # Record contagion event
                contagion_event = BiasContagionEvent(
                    source_agent=source_agent.agent_id,
                    target_agent=target_agent.agent_id,
                    transmitted_trait=trait_to_transmit,
                    generation=self.generation,
                    interaction_round=len(target_agent.memories),
                    transmission_strength=transmission_strength,
                    timestamp=datetime.now()
                )
                self.contagion_events.append(contagion_event)
                
                self.logger.debug(f"Contagion: {source_agent.agent_id} -> {target_agent.agent_id}, trait: {trait_to_transmit}")
    
    def _select_trait_for_transmission(self, source_profile, target_profile) -> Optional[str]:
        """Select which psychological trait should be transmitted"""
        
        # Prioritize traits that are significantly different
        trait_candidates = []
        
        if abs(source_profile.trust_level - target_profile.trust_level) > 0.2:
            trait_candidates.append("trust_level")
        
        if abs(source_profile.loss_sensitivity - target_profile.loss_sensitivity) > 0.3:
            trait_candidates.append("loss_sensitivity")
        
        if abs(source_profile.risk_tolerance - target_profile.risk_tolerance) > 0.2:
            trait_candidates.append("risk_tolerance")
        
        # Also consider transmitting learned heuristics
        unique_heuristics = set(source_profile.learned_heuristics) - set(target_profile.learned_heuristics)
        if unique_heuristics:
            trait_candidates.append("heuristics")
        
        return random.choice(trait_candidates) if trait_candidates else None
    
    def _apply_trait_transmission(self, target_profile, source_profile, trait: str, strength: float) -> None:
        """Apply the actual trait transmission"""
        
        if trait == "trust_level":
            target_profile.trust_level += (source_profile.trust_level - target_profile.trust_level) * strength
            target_profile.trust_level = max(0.0, min(1.0, target_profile.trust_level))
        
        elif trait == "loss_sensitivity":
            target_profile.loss_sensitivity += (source_profile.loss_sensitivity - target_profile.loss_sensitivity) * strength
            target_profile.loss_sensitivity = max(0.5, min(3.0, target_profile.loss_sensitivity))
        
        elif trait == "risk_tolerance":
            target_profile.risk_tolerance += (source_profile.risk_tolerance - target_profile.risk_tolerance) * strength
            target_profile.risk_tolerance = max(0.0, min(1.0, target_profile.risk_tolerance))
        
        elif trait == "heuristics":
            # Randomly select a heuristic to transmit
            available_heuristics = list(set(source_profile.learned_heuristics) - set(target_profile.learned_heuristics))
            if available_heuristics:
                transmitted_heuristic = random.choice(available_heuristics)
                target_profile.learned_heuristics.append(transmitted_heuristic)
    
    def _run_evolution_phase(self) -> None:
        """Evolution phase where successful psychological profiles reproduce"""
        
        # Identify most and least successful agents
        sorted_agents = sorted(self.population, key=lambda a: a.total_score, reverse=True)
        
        # Top 10% serve as "parents" for psychological traits
        elite_count = max(1, self.population_size // 10)
        elite_agents = sorted_agents[:elite_count]
        
        # Bottom 10% get "reset" with traits from elite agents
        struggling_count = max(1, self.population_size // 10)
        struggling_agents = sorted_agents[-struggling_count:]
        
        for struggling_agent in struggling_agents:
            # Select random elite agent as "parent"
            parent_agent = random.choice(elite_agents)
            
            # Copy psychological traits with some mutation
            self._inherit_psychological_traits(struggling_agent, parent_agent)
            
            self.logger.debug(f"Evolution: {struggling_agent.agent_id} inherited traits from {parent_agent.agent_id}")
    
    def _inherit_psychological_traits(self, child_agent: EmergentBiasAgent, parent_agent: EmergentBiasAgent) -> None:
        """Copy psychological traits from parent to child with mutation"""
        
        parent_profile = parent_agent.psychological_profile
        child_profile = child_agent.psychological_profile
        
        # Inherit core traits with small random mutations
        mutation_factor = 0.1
        
        child_profile.trust_level = parent_profile.trust_level + random.uniform(-mutation_factor, mutation_factor)
        child_profile.trust_level = max(0.0, min(1.0, child_profile.trust_level))
        
        child_profile.loss_sensitivity = parent_profile.loss_sensitivity + random.uniform(-mutation_factor, mutation_factor)
        child_profile.loss_sensitivity = max(0.5, min(3.0, child_profile.loss_sensitivity))
        
        child_profile.risk_tolerance = parent_profile.risk_tolerance + random.uniform(-mutation_factor, mutation_factor)
        child_profile.risk_tolerance = max(0.0, min(1.0, child_profile.risk_tolerance))
        
        # Inherit some learned heuristics
        child_profile.learned_heuristics = parent_profile.learned_heuristics.copy()
        
        # Clear trauma memories (fresh start) but keep some wisdom
        child_profile.trauma_memories.clear()
        child_profile.emotional_state = "hopeful"
        
        # Reset scores but keep psychological learning
        child_agent.total_score = 0
        child_agent.round_count = 0
        child_agent.memories.clear()
    
    def _take_population_snapshot(self) -> PopulationSnapshot:
        """Take snapshot of current population state"""
        
        # Calculate psychological distribution
        trait_counts = {}
        trust_levels = []
        loss_sensitivities = []
        cooperation_rates = []
        
        for agent in self.population:
            profile = agent.psychological_profile
            dominant_trait = profile.get_dominant_trait()
            
            trait_counts[dominant_trait] = trait_counts.get(dominant_trait, 0) + 1
            trust_levels.append(profile.trust_level)
            loss_sensitivities.append(profile.loss_sensitivity)
            cooperation_rates.append(agent.get_cooperation_rate())
        
        # Identify dominant traits (>10% of population)
        threshold = self.population_size * 0.1
        dominant_traits = [trait for trait, count in trait_counts.items() if count >= threshold]
        
        return PopulationSnapshot(
            generation=self.generation,
            population_size=self.population_size,
            psychological_distribution=trait_counts,
            avg_trust_level=np.mean(trust_levels),
            avg_loss_sensitivity=np.mean(loss_sensitivities),
            avg_cooperation_rate=np.mean(cooperation_rates),
            dominant_traits=dominant_traits,
            timestamp=datetime.now()
        )
    
    def _log_generation_summary(self, snapshot: PopulationSnapshot) -> None:
        """Log summary of generation"""
        
        self.logger.info(f"Generation {snapshot.generation} Summary:")
        self.logger.info(f"  Avg Trust Level: {snapshot.avg_trust_level:.3f}")
        self.logger.info(f"  Avg Loss Sensitivity: {snapshot.avg_loss_sensitivity:.3f}")
        self.logger.info(f"  Avg Cooperation Rate: {snapshot.avg_cooperation_rate:.3f}")
        self.logger.info(f"  Dominant Traits: {snapshot.dominant_traits}")
        self.logger.info(f"  Trait Distribution: {snapshot.psychological_distribution}")
    
    def analyze_population_evolution(self) -> Dict[str, Any]:
        """Analyze how population psychology evolved over time"""
        
        if len(self.population_history) < 2:
            return {"error": "Insufficient data for evolution analysis"}
        
        first_snapshot = self.population_history[0]
        last_snapshot = self.population_history[-1]
        
        # Calculate changes
        trust_change = last_snapshot.avg_trust_level - first_snapshot.avg_trust_level
        loss_sensitivity_change = last_snapshot.avg_loss_sensitivity - first_snapshot.avg_loss_sensitivity
        cooperation_change = last_snapshot.avg_cooperation_rate - first_snapshot.avg_cooperation_rate
        
        # Analyze trait evolution
        trait_evolution = {}
        for snapshot in self.population_history:
            for trait, count in snapshot.psychological_distribution.items():
                if trait not in trait_evolution:
                    trait_evolution[trait] = []
                trait_evolution[trait].append(count)
        
        # Find traits that emerged or died out
        final_traits = set(last_snapshot.psychological_distribution.keys())
        initial_traits = set(first_snapshot.psychological_distribution.keys())
        
        emerged_traits = final_traits - initial_traits
        extinct_traits = initial_traits - final_traits
        
        # Analyze contagion patterns
        contagion_analysis = self._analyze_contagion_patterns()
        
        return {
            "total_generations": len(self.population_history),
            "psychological_changes": {
                "trust_level_change": trust_change,
                "loss_sensitivity_change": loss_sensitivity_change,
                "cooperation_rate_change": cooperation_change
            },
            "trait_evolution": trait_evolution,
            "emerged_traits": list(emerged_traits),
            "extinct_traits": list(extinct_traits),
            "final_dominant_traits": last_snapshot.dominant_traits,
            "contagion_analysis": contagion_analysis,
            "population_snapshots": self.population_history
        }
    
    def _analyze_contagion_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in psychological contagion"""
        
        if not self.contagion_events:
            return {"total_events": 0}
        
        # Count transmissions by trait type
        trait_transmissions = {}
        generation_transmissions = {}
        
        for event in self.contagion_events:
            trait_transmissions[event.transmitted_trait] = trait_transmissions.get(event.transmitted_trait, 0) + 1
            generation_transmissions[event.generation] = generation_transmissions.get(event.generation, 0) + 1
        
        # Find super-spreaders (agents who transmitted many traits)
        spreader_counts = {}
        for event in self.contagion_events:
            spreader_counts[event.source_agent] = spreader_counts.get(event.source_agent, 0) + 1
        
        super_spreaders = [(agent, count) for agent, count in spreader_counts.items() if count >= 3]
        super_spreaders.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_events": len(self.contagion_events),
            "trait_transmission_counts": trait_transmissions,
            "most_transmitted_trait": max(trait_transmissions.items(), key=lambda x: x[1])[0] if trait_transmissions else None,
            "super_spreaders": super_spreaders[:5],  # Top 5
            "transmission_by_generation": generation_transmissions,
            "avg_transmission_strength": np.mean([e.transmission_strength for e in self.contagion_events])
        }
    
    def get_agent_psychological_trajectories(self) -> Dict[str, List[Dict]]:
        """Get psychological evolution trajectory for each agent"""
        
        trajectories = {}
        
        for agent in self.population:
            # This would require storing historical psychological states
            # For now, return current state
            trajectories[agent.agent_id] = [agent.get_psychological_summary()]
        
        return trajectories
    
    def export_population_data(self, filepath: str) -> None:
        """Export complete population data for analysis"""
        
        export_data = {
            "population_size": self.population_size,
            "total_generations": self.generation,
            "population_history": [
                {
                    "generation": s.generation,
                    "psychological_distribution": s.psychological_distribution,
                    "avg_trust_level": s.avg_trust_level,
                    "avg_loss_sensitivity": s.avg_loss_sensitivity,
                    "avg_cooperation_rate": s.avg_cooperation_rate,
                    "dominant_traits": s.dominant_traits,
                    "timestamp": s.timestamp.isoformat()
                }
                for s in self.population_history
            ],
            "contagion_events": [
                {
                    "source_agent": e.source_agent,
                    "target_agent": e.target_agent,
                    "transmitted_trait": e.transmitted_trait,
                    "generation": e.generation,
                    "transmission_strength": e.transmission_strength,
                    "timestamp": e.timestamp.isoformat()
                }
                for e in self.contagion_events
            ],
            "final_agent_states": [
                agent.get_psychological_summary() for agent in self.population
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Population data exported to {filepath}")
    
    def create_psychological_lineage_tree(self) -> Dict[str, Any]:
        """Create tree showing how psychological traits spread through population"""
        
        # This would track which agents influenced which others
        # For now, return a simplified version based on contagion events
        
        influence_network = {}
        
        for event in self.contagion_events:
            source = event.source_agent
            target = event.target_agent
            
            if source not in influence_network:
                influence_network[source] = {"influenced": [], "traits_spread": []}
            
            influence_network[source]["influenced"].append(target)
            influence_network[source]["traits_spread"].append(event.transmitted_trait)
        
        return {
            "influence_network": influence_network,
            "most_influential_agents": sorted(
                [(agent, len(data["influenced"])) for agent, data in influence_network.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }