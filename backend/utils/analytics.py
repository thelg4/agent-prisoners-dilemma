# backend/utils/analytics.py
"""
Analytics and statistics utilities for game analysis
"""

from typing import List, Dict, Any, Optional, Tuple
import statistics
import math
from datetime import datetime, timedelta

from ..game.models import GameState, GameResult, Move, AgentType


class GameAnalytics:
    """Comprehensive analytics for prisoner's dilemma games"""
    
    @staticmethod
    def calculate_cooperation_trends(game_history: List[GameResult], window_size: int = 3) -> Dict[str, List[float]]:
        """Calculate cooperation rate trends over time using moving averages"""
        if len(game_history) < window_size:
            return {"agent1_trend": [], "agent2_trend": []}
        
        agent1_trend = []
        agent2_trend = []
        
        for i in range(window_size - 1, len(game_history)):
            window = game_history[i - window_size + 1:i + 1]
            
            agent1_coops = sum(1 for r in window if r.agent1_move == Move.COOPERATE)
            agent2_coops = sum(1 for r in window if r.agent2_move == Move.COOPERATE)
            
            agent1_trend.append(agent1_coops / window_size)
            agent2_trend.append(agent2_coops / window_size)
        
        return {
            "agent1_trend": agent1_trend,
            "agent2_trend": agent2_trend,
            "window_size": window_size
        }
    
    @staticmethod
    def analyze_strategy_patterns(game_history: List[GameResult]) -> Dict[str, Any]:
        """Analyze strategic patterns in agent behavior"""
        if len(game_history) < 2:
            return {"insufficient_data": True}
        
        patterns = {
            "agent1": GameAnalytics._analyze_agent_patterns([r.agent1_move for r in game_history]),
            "agent2": GameAnalytics._analyze_agent_patterns([r.agent2_move for r in game_history])
        }
        
        # Analyze interaction patterns
        patterns["interactions"] = GameAnalytics._analyze_interaction_patterns(game_history)
        
        return patterns
    
    @staticmethod
    def _analyze_agent_patterns(moves: List[Move]) -> Dict[str, Any]:
        """Analyze patterns for a single agent"""
        if len(moves) < 3:
            return {"insufficient_data": True}
        
        # Calculate streaks
        current_streak = 1
        max_coop_streak = 0
        max_defect_streak = 0
        coop_streaks = []
        defect_streaks = []
        
        for i in range(1, len(moves)):
            if moves[i] == moves[i-1]:
                current_streak += 1
            else:
                if moves[i-1] == Move.COOPERATE:
                    coop_streaks.append(current_streak)
                    max_coop_streak = max(max_coop_streak, current_streak)
                else:
                    defect_streaks.append(current_streak)
                    max_defect_streak = max(max_defect_streak, current_streak)
                current_streak = 1
        
        # Handle final streak
        if moves[-1] == Move.COOPERATE:
            coop_streaks.append(current_streak)
            max_coop_streak = max(max_coop_streak, current_streak)
        else:
            defect_streaks.append(current_streak)
            max_defect_streak = max(max_defect_streak, current_streak)
        
        # Check for alternating pattern
        alternating_count = 0
        for i in range(1, len(moves)):
            if moves[i] != moves[i-1]:
                alternating_count += 1
        
        is_alternating = alternating_count / (len(moves) - 1) > 0.7 if len(moves) > 1 else False
        
        # Check for tit-for-tat like behavior (would need opponent moves for full analysis)
        return {
            "max_cooperation_streak": max_coop_streak,
            "max_defection_streak": max_defect_streak,
            "average_cooperation_streak": statistics.mean(coop_streaks) if coop_streaks else 0,
            "average_defection_streak": statistics.mean(defect_streaks) if defect_streaks else 0,
            "is_alternating": is_alternating,
            "alternating_rate": alternating_count / (len(moves) - 1) if len(moves) > 1 else 0,
            "consistency_score": GameAnalytics._calculate_consistency(moves)
        }
    
    @staticmethod
    def _analyze_interaction_patterns(game_history: List[GameResult]) -> Dict[str, Any]:
        """Analyze how agents respond to each other"""
        if len(game_history) < 2:
            return {"insufficient_data": True}
        
        # Tit-for-tat analysis
        agent1_reciprocity = 0
        agent2_reciprocity = 0
        
        # Mutual cooperation/defection analysis
        mutual_coop_runs = []
        mutual_defect_runs = []
        current_mutual_coop = 0
        current_mutual_defect = 0
        
        for i, result in enumerate(game_history):
            # Check reciprocity (responding to opponent's previous move)
            if i > 0:
                prev_result = game_history[i-1]
                if result.agent1_move == prev_result.agent2_move:
                    agent1_reciprocity += 1
                if result.agent2_move == prev_result.agent1_move:
                    agent2_reciprocity += 1
            
            # Track mutual cooperation/defection runs
            if result.agent1_move == Move.COOPERATE and result.agent2_move == Move.COOPERATE:
                current_mutual_coop += 1
                if current_mutual_defect > 0:
                    mutual_defect_runs.append(current_mutual_defect)
                    current_mutual_defect = 0
            elif result.agent1_move == Move.DEFECT and result.agent2_move == Move.DEFECT:
                current_mutual_defect += 1
                if current_mutual_coop > 0:
                    mutual_coop_runs.append(current_mutual_coop)
                    current_mutual_coop = 0
            else:
                if current_mutual_coop > 0:
                    mutual_coop_runs.append(current_mutual_coop)
                    current_mutual_coop = 0
                if current_mutual_defect > 0:
                    mutual_defect_runs.append(current_mutual_defect)
                    current_mutual_defect = 0
        
        # Add final runs
        if current_mutual_coop > 0:
            mutual_coop_runs.append(current_mutual_coop)
        if current_mutual_defect > 0:
            mutual_defect_runs.append(current_mutual_defect)
        
        return {
            "agent1_reciprocity_rate": agent1_reciprocity / (len(game_history) - 1) if len(game_history) > 1 else 0,
            "agent2_reciprocity_rate": agent2_reciprocity / (len(game_history) - 1) if len(game_history) > 1 else 0,
            "mutual_cooperation_runs": mutual_coop_runs,
            "mutual_defection_runs": mutual_defect_runs,
            "average_mutual_coop_length": statistics.mean(mutual_coop_runs) if mutual_coop_runs else 0,
            "average_mutual_defect_length": statistics.mean(mutual_defect_runs) if mutual_defect_runs else 0,
            "longest_mutual_cooperation": max(mutual_coop_runs) if mutual_coop_runs else 0,
            "longest_mutual_defection": max(mutual_defect_runs) if mutual_defect_runs else 0
        }
    
    @staticmethod
    def _calculate_consistency(moves: List[Move]) -> float:
        """Calculate how consistent an agent's strategy is"""
        if len(moves) < 2:
            return 1.0
        
        # Simple consistency: how often they stick with the same move
        same_moves = sum(1 for i in range(1, len(moves)) if moves[i] == moves[i-1])
        return same_moves / (len(moves) - 1)
    
    @staticmethod
    def calculate_loss_aversion_impact(game_state: GameState) -> Dict[str, Any]:
        """Calculate the impact of loss aversion on the game"""
        game_history = game_state["game_history"]
        loss_factor = game_state["loss_aversion_factor"]
        reference_point = game_state["reference_point"]
        
        if not game_history:
            return {"no_data": True}
        
        agent1_losses = 0
        agent2_losses = 0
        agent1_total_penalty = 0
        agent2_total_penalty = 0
        
        # Analyze each round
        for result in game_history:
            # Count losses (payoffs below reference point)
            if result.agent1_payoff < reference_point:
                agent1_losses += 1
                agent1_total_penalty += (result.agent1_payoff - result.agent1_adjusted_payoff)
            
            if result.agent2_payoff < reference_point:
                agent2_losses += 1
                agent2_total_penalty += (result.agent2_payoff - result.agent2_adjusted_payoff)
        
        total_rounds = len(game_history)
        
        # Calculate impact on final outcome
        raw_winner = None
        if game_state["agent1"].total_payoff > game_state["agent2"].total_payoff:
            raw_winner = "agent1"
        elif game_state["agent2"].total_payoff > game_state["agent1"].total_payoff:
            raw_winner = "agent2"
        else:
            raw_winner = "tie"
        
        actual_winner = game_state.get("winner", "unknown")
        outcome_changed = raw_winner != actual_winner
        
        return {
            "loss_frequency": {
                "agent1": agent1_losses,
                "agent2": agent2_losses,
                "agent1_rate": agent1_losses / total_rounds,
                "agent2_rate": agent2_losses / total_rounds
            },
            "loss_penalties": {
                "agent1_total": agent1_total_penalty,
                "agent2_total": agent2_total_penalty,
                "agent1_average": agent1_total_penalty / total_rounds,
                "agent2_average": agent2_total_penalty / total_rounds
            },
            "outcome_impact": {
                "raw_winner": raw_winner,
                "actual_winner": actual_winner,
                "outcome_changed": outcome_changed
            },
            "settings": {
                "loss_aversion_factor": loss_factor,
                "reference_point": reference_point
            }
        }
    
    @staticmethod
    def generate_behavioral_insights(game_state: GameState) -> Dict[str, Any]:
        """Generate high-level behavioral insights"""
        game_history = game_state["game_history"]
        agent1_state = game_state["agent1"]
        agent2_state = game_state["agent2"]
        
        if not game_history:
            return {"no_data": True}
        
        insights = {}
        
        # Strategy classification
        insights["strategy_classification"] = {
            "agent1": GameAnalytics._classify_strategy(agent1_state, [r.agent2_move for r in game_history]),
            "agent2": GameAnalytics._classify_strategy(agent2_state, [r.agent1_move for r in game_history])
        }
        
        # Performance analysis
        insights["performance"] = {
            "agent1": {
                "efficiency": agent1_state.adjusted_total_payoff / (len(game_history) * 5),  # Max possible per round is 5
                "cooperation_effectiveness": GameAnalytics._calculate_cooperation_effectiveness(
                    [r.agent1_move for r in game_history],
                    [r.agent1_payoff for r in game_history]
                )
            },
            "agent2": {
                "efficiency": agent2_state.adjusted_total_payoff / (len(game_history) * 5),
                "cooperation_effectiveness": GameAnalytics._calculate_cooperation_effectiveness(
                    [r.agent2_move for r in game_history],
                    [r.agent2_payoff for r in game_history]
                )
            }
        }
        
        # Learning patterns
        insights["learning_patterns"] = GameAnalytics._analyze_learning_patterns(game_history)
        
        # Behavioral economics insights
        if agent1_state.agent_type == AgentType.HUMAN or agent2_state.agent_type == AgentType.HUMAN:
            insights["behavioral_economics"] = GameAnalytics._analyze_behavioral_economics(game_state)
        
        return insights
    
    @staticmethod
    def _classify_strategy(agent_state, opponent_moves: List[Move]) -> str:
        """Classify an agent's overall strategy"""
        coop_rate = agent_state.cooperation_rate
        moves = agent_state.moves_history
        
        if not moves or not opponent_moves:
            return "unknown"
        
        # Always cooperate/defect
        if coop_rate >= 0.9:
            return "always_cooperate"
        elif coop_rate <= 0.1:
            return "always_defect"
        
        # Check for tit-for-tat pattern
        if len(moves) > 1 and len(opponent_moves) >= len(moves) - 1:
            tit_for_tat_matches = 0
            for i in range(1, len(moves)):
                if i-1 < len(opponent_moves) and moves[i] == opponent_moves[i-1]:
                    tit_for_tat_matches += 1
            
            tit_for_tat_rate = tit_for_tat_matches / (len(moves) - 1)
            if tit_for_tat_rate >= 0.7:
                return "tit_for_tat"
        
        # Check for alternating
        alternating = 0
        for i in range(1, len(moves)):
            if moves[i] != moves[i-1]:
                alternating += 1
        
        if alternating / (len(moves) - 1) >= 0.7:
            return "alternating"
        
        # Classify by cooperation rate
        if 0.6 <= coop_rate <= 0.8:
            return "mostly_cooperative"
        elif 0.2 <= coop_rate <= 0.4:
            return "mostly_defective"
        else:
            return "mixed_strategy"
    
    @staticmethod
    def _calculate_cooperation_effectiveness(moves: List[Move], payoffs: List[float]) -> float:
        """Calculate how effective cooperation was for an agent"""
        if not moves or not payoffs:
            return 0.0
        
        coop_payoffs = [payoffs[i] for i, move in enumerate(moves) if move == Move.COOPERATE]
        defect_payoffs = [payoffs[i] for i, move in enumerate(moves) if move == Move.DEFECT]
        
        avg_coop_payoff = statistics.mean(coop_payoffs) if coop_payoffs else 0
        avg_defect_payoff = statistics.mean(defect_payoffs) if defect_payoffs else 0
        
        # Return ratio of cooperation effectiveness
        if avg_defect_payoff == 0:
            return float('inf') if avg_coop_payoff > 0 else 0
        
        return avg_coop_payoff / avg_defect_payoff
    
    @staticmethod
    def _analyze_learning_patterns(game_history: List[GameResult]) -> Dict[str, Any]:
        """Analyze if agents show learning behavior over time"""
        if len(game_history) < 6:
            return {"insufficient_data": True}
        
        # Split into early and late phases
        mid_point = len(game_history) // 2
        early_phase = game_history[:mid_point]
        late_phase = game_history[mid_point:]
        
        # Calculate cooperation rates for each phase
        early_agent1_coop = sum(1 for r in early_phase if r.agent1_move == Move.COOPERATE) / len(early_phase)
        late_agent1_coop = sum(1 for r in late_phase if r.agent1_move == Move.COOPERATE) / len(late_phase)
        
        early_agent2_coop = sum(1 for r in early_phase if r.agent2_move == Move.COOPERATE) / len(early_phase)
        late_agent2_coop = sum(1 for r in late_phase if r.agent2_move == Move.COOPERATE) / len(late_phase)
        
        # Calculate mutual cooperation rates
        early_mutual_coop = sum(1 for r in early_phase 
                               if r.agent1_move == Move.COOPERATE and r.agent2_move == Move.COOPERATE) / len(early_phase)
        late_mutual_coop = sum(1 for r in late_phase 
                              if r.agent1_move == Move.COOPERATE and r.agent2_move == Move.COOPERATE) / len(late_phase)
        
        return {
            "phase_comparison": {
                "agent1_cooperation_change": late_agent1_coop - early_agent1_coop,
                "agent2_cooperation_change": late_agent2_coop - early_agent2_coop,
                "mutual_cooperation_change": late_mutual_coop - early_mutual_coop
            },
            "learning_indicators": {
                "agent1_adapted": abs(late_agent1_coop - early_agent1_coop) > 0.2,
                "agent2_adapted": abs(late_agent2_coop - early_agent2_coop) > 0.2,
                "system_stabilized": abs(late_mutual_coop - early_mutual_coop) < 0.1
            }
        }
    
    @staticmethod
    def _analyze_behavioral_economics(game_state: GameState) -> Dict[str, Any]:
        """Analyze behavioral economics aspects"""
        game_history = game_state["game_history"]
        
        # Find instances where loss aversion likely influenced decisions
        loss_aversion_events = []
        
        for i, result in enumerate(game_history):
            # Look for cases where an agent got the "sucker's payoff"
            if result.agent1_move == Move.COOPERATE and result.agent2_move == Move.DEFECT:
                loss_aversion_events.append({
                    "round": i + 1,
                    "victim": "agent1",
                    "type": "sucker_payoff"
                })
            elif result.agent1_move == Move.DEFECT and result.agent2_move == Move.COOPERATE:
                loss_aversion_events.append({
                    "round": i + 1,
                    "victim": "agent2",
                    "type": "sucker_payoff"
                })
        
        # Analyze behavioral responses to loss aversion events
        behavioral_responses = []
        for event in loss_aversion_events:
            round_num = event["round"]
            if round_num < len(game_history):  # Check next round response
                next_result = game_history[round_num]  # round_num is 1-indexed
                victim = event["victim"]
                
                if victim == "agent1":
                    response = "defected" if next_result.agent1_move == Move.DEFECT else "cooperated"
                else:
                    response = "defected" if next_result.agent2_move == Move.DEFECT else "cooperated"
                
                behavioral_responses.append({
                    "after_round": round_num,
                    "victim": victim,
                    "response": response
                })
        
        return {
            "loss_aversion_events": loss_aversion_events,
            "behavioral_responses": behavioral_responses,
            "loss_aversion_triggered_defection": sum(1 for r in behavioral_responses if r["response"] == "defected"),
            "total_sucker_payoffs": len(loss_aversion_events)
        }


class GameStatistics:
    """Statistical analysis utilities for multiple games"""
    
    @staticmethod
    def aggregate_game_stats(games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate statistics across multiple games"""
        if not games:
            return {"no_data": True}
        
        total_games = len(games)
        
        # Win rates by agent type
        econ_wins = 0
        human_wins = 0
        ties = 0
        
        # Cooperation rates
        econ_coop_rates = []
        human_coop_rates = []
        
        # Average payoffs
        econ_payoffs = []
        human_payoffs = []
        
        # Loss aversion impacts
        loss_aversion_impacts = []
        
        for game in games:
            game_summary = game.get("game_summary", {})
            agents = game_summary.get("agents", {})
            
            # Determine winner
            winner = game_summary.get("winner")
            if winner == "tie":
                ties += 1
            else:
                # Check agent types to determine if econ or human won
                agent1 = agents.get("agent1", {})
                agent2 = agents.get("agent2", {})
                
                if winner == "agent1":
                    if agent1.get("type") == "econ":
                        econ_wins += 1
                    else:
                        human_wins += 1
                elif winner == "agent2":
                    if agent2.get("type") == "econ":
                        econ_wins += 1
                    else:
                        human_wins += 1
            
            # Collect cooperation rates and payoffs by type
            for agent_key, agent_data in agents.items():
                if agent_data.get("type") == "econ":
                    econ_coop_rates.append(agent_data.get("cooperation_rate", 0))
                    econ_payoffs.append(agent_data.get("adjusted_payoff", 0))
                elif agent_data.get("type") == "human":
                    human_coop_rates.append(agent_data.get("cooperation_rate", 0))
                    human_payoffs.append(agent_data.get("adjusted_payoff", 0))
                    
                    # Collect loss aversion impact
                    impact = agent_data.get("loss_aversion_impact", 0)
                    if impact != 0:
                        loss_aversion_impacts.append(abs(impact))
        
        return {
            "total_games": total_games,
            "win_rates": {
                "econ_agent": econ_wins / total_games if total_games > 0 else 0,
                "human_agent": human_wins / total_games if total_games > 0 else 0,
                "ties": ties / total_games if total_games > 0 else 0
            },
            "cooperation_rates": {
                "econ_average": statistics.mean(econ_coop_rates) if econ_coop_rates else 0,
                "human_average": statistics.mean(human_coop_rates) if human_coop_rates else 0,
                "econ_std": statistics.stdev(econ_coop_rates) if len(econ_coop_rates) > 1 else 0,
                "human_std": statistics.stdev(human_coop_rates) if len(human_coop_rates) > 1 else 0
            },
            "payoff_performance": {
                "econ_average": statistics.mean(econ_payoffs) if econ_payoffs else 0,
                "human_average": statistics.mean(human_payoffs) if human_payoffs else 0,
                "econ_std": statistics.stdev(econ_payoffs) if len(econ_payoffs) > 1 else 0,
                "human_std": statistics.stdev(human_payoffs) if len(human_payoffs) > 1 else 0
            },
            "loss_aversion": {
                "average_impact": statistics.mean(loss_aversion_impacts) if loss_aversion_impacts else 0,
                "max_impact": max(loss_aversion_impacts) if loss_aversion_impacts else 0,
                "games_with_impact": len([i for i in loss_aversion_impacts if i > 0])
            }
        }
    
    @staticmethod
    def compare_agent_types(games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance between different agent types"""
        if not games:
            return {"no_data": True}
        
        # Separate games by matchup type
        econ_vs_human = []
        econ_vs_econ = []
        human_vs_human = []
        
        for game in games:
            game_summary = game.get("game_summary", {})
            agents = game_summary.get("agents", {})
            
            agent1_type = agents.get("agent1", {}).get("type")
            agent2_type = agents.get("agent2", {}).get("type")
            
            if agent1_type == "econ" and agent2_type == "human":
                econ_vs_human.append(game)
            elif agent1_type == "human" and agent2_type == "econ":
                econ_vs_human.append(game)
            elif agent1_type == "econ" and agent2_type == "econ":
                econ_vs_econ.append(game)
            elif agent1_type == "human" and agent2_type == "human":
                human_vs_human.append(game)
        
        return {
            "matchup_analysis": {
                "econ_vs_human": {
                    "total_games": len(econ_vs_human),
                    "stats": GameStatistics.aggregate_game_stats(econ_vs_human)
                },
                "econ_vs_econ": {
                    "total_games": len(econ_vs_econ),
                    "stats": GameStatistics.aggregate_game_stats(econ_vs_econ)
                },
                "human_vs_human": {
                    "total_games": len(human_vs_human),
                    "stats": GameStatistics.aggregate_game_stats(human_vs_human)
                }
            }
        }
    
    @staticmethod
    def analyze_loss_aversion_effectiveness(games: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how different loss aversion factors affect outcomes"""
        if not games:
            return {"no_data": True}
        
        # Group games by loss aversion factor
        factor_groups = {}
        
        for game in games:
            game_info = game.get("game_summary", {}).get("game_info", {})
            factor = game_info.get("loss_aversion_factor", 2.0)
            
            if factor not in factor_groups:
                factor_groups[factor] = []
            factor_groups[factor].append(game)
        
        # Analyze each group
        factor_analysis = {}
        for factor, factor_games in factor_groups.items():
            factor_analysis[str(factor)] = {
                "game_count": len(factor_games),
                "stats": GameStatistics.aggregate_game_stats(factor_games)
            }
        
        return {
            "factor_analysis": factor_analysis,
            "factor_range": {
                "min": min(factor_groups.keys()) if factor_groups else 0,
                "max": max(factor_groups.keys()) if factor_groups else 0,
                "factors_tested": list(factor_groups.keys())
            }
        }


class RealtimeAnalytics:
    """Real-time analytics for ongoing games"""
    
    @staticmethod
    def get_current_game_insights(game_state: GameState) -> Dict[str, Any]:
        """Get real-time insights for a game in progress"""
        game_history = game_state["game_history"]
        current_round = game_state["current_round"]
        
        if not game_history:
            return {
                "status": "waiting_for_first_round",
                "predictions": RealtimeAnalytics._make_initial_predictions(game_state)
            }
        
        insights = {
            "current_status": {
                "round": current_round,
                "total_rounds": len(game_history),
                "progress": len(game_history) / game_state["max_rounds"]
            },
            "recent_trends": GameAnalytics.calculate_cooperation_trends(game_history, window_size=3),
            "momentum": RealtimeAnalytics._calculate_momentum(game_history),
            "predictions": RealtimeAnalytics._make_predictions(game_state)
        }
        
        # Add loss aversion insights if applicable
        if game_state["agent1"].agent_type == AgentType.HUMAN or game_state["agent2"].agent_type == AgentType.HUMAN:
            insights["loss_aversion_tracking"] = RealtimeAnalytics._track_loss_aversion_effects(game_state)
        
        return insights
    
    @staticmethod
    def _calculate_momentum(game_history: List[GameResult], window_size: int = 3) -> Dict[str, Any]:
        """Calculate current momentum in cooperation/defection"""
        if len(game_history) < window_size:
            return {"insufficient_data": True}
        
        recent_rounds = game_history[-window_size:]
        
        agent1_recent_coop = sum(1 for r in recent_rounds if r.agent1_move == Move.COOPERATE)
        agent2_recent_coop = sum(1 for r in recent_rounds if r.agent2_move == Move.COOPERATE)
        
        mutual_coop_recent = sum(1 for r in recent_rounds 
                                if r.agent1_move == Move.COOPERATE and r.agent2_move == Move.COOPERATE)
        
        return {
            "agent1_cooperation_momentum": agent1_recent_coop / window_size,
            "agent2_cooperation_momentum": agent2_recent_coop / window_size,
            "mutual_cooperation_momentum": mutual_coop_recent / window_size,
            "window_size": window_size
        }
    
    @staticmethod
    def _make_initial_predictions(game_state: GameState) -> Dict[str, Any]:
        """Make predictions before the game starts based on agent types"""
        agent1_type = game_state["agent1"].agent_type
        agent2_type = game_state["agent2"].agent_type
        loss_factor = game_state["loss_aversion_factor"]
        
        predictions = {
            "likely_winner": "unknown",
            "expected_cooperation_rate": 0.5,
            "expected_mutual_cooperation": 0.3,
            "confidence": 0.3
        }
        
        # Adjust predictions based on agent types
        if agent1_type == AgentType.ECON and agent2_type == AgentType.HUMAN:
            predictions.update({
                "likely_winner": "agent1",
                "expected_cooperation_rate": 0.4,
                "expected_mutual_cooperation": 0.2,
                "confidence": 0.6,
                "reasoning": "Econ agent likely to exploit human agent's loss aversion"
            })
        elif agent1_type == AgentType.HUMAN and agent2_type == AgentType.ECON:
            predictions.update({
                "likely_winner": "agent2",
                "expected_cooperation_rate": 0.4,
                "expected_mutual_cooperation": 0.2,
                "confidence": 0.6,
                "reasoning": "Econ agent likely to exploit human agent's loss aversion"
            })
        elif agent1_type == AgentType.HUMAN and agent2_type == AgentType.HUMAN:
            predictions.update({
                "likely_winner": "tie",
                "expected_cooperation_rate": 0.6,
                "expected_mutual_cooperation": 0.4,
                "confidence": 0.4,
                "reasoning": "Both agents subject to loss aversion, may cooperate more"
            })
        elif agent1_type == AgentType.ECON and agent2_type == AgentType.ECON:
            predictions.update({
                "likely_winner": "tie",
                "expected_cooperation_rate": 0.3,
                "expected_mutual_cooperation": 0.1,
                "confidence": 0.7,
                "reasoning": "Both rational agents likely to defect frequently"
            })
        
        # Adjust for loss aversion factor
        if loss_factor > 2.5:
            predictions["expected_cooperation_rate"] *= 0.8
            predictions["expected_mutual_cooperation"] *= 0.7
        
        return predictions
    
    @staticmethod
    def _make_predictions(game_state: GameState) -> Dict[str, Any]:
        """Make predictions based on current game state"""
        game_history = game_state["game_history"]
        agent1_state = game_state["agent1"]
        agent2_state = game_state["agent2"]
        rounds_remaining = game_state["max_rounds"] - len(game_history)
        
        if rounds_remaining <= 0:
            return {"game_complete": True}
        
        # Predict likely winner based on current trajectory
        current_diff = agent1_state.adjusted_total_payoff - agent2_state.adjusted_total_payoff
        
        # Simple projection based on recent performance
        recent_rounds = min(5, len(game_history))
        if recent_rounds > 0:
            recent_history = game_history[-recent_rounds:]
            agent1_recent_avg = sum(r.agent1_adjusted_payoff for r in recent_history) / recent_rounds
            agent2_recent_avg = sum(r.agent2_adjusted_payoff for r in recent_history) / recent_rounds
            
            projected_agent1 = agent1_state.adjusted_total_payoff + (agent1_recent_avg * rounds_remaining)
            projected_agent2 = agent2_state.adjusted_total_payoff + (agent2_recent_avg * rounds_remaining)
            
            if projected_agent1 > projected_agent2:
                likely_winner = "agent1"
                confidence = min(0.9, abs(projected_agent1 - projected_agent2) / 10)
            elif projected_agent2 > projected_agent1:
                likely_winner = "agent2"
                confidence = min(0.9, abs(projected_agent1 - projected_agent2) / 10)
            else:
                likely_winner = "tie"
                confidence = 0.3
        else:
            likely_winner = "unknown"
            confidence = 0.1
        
        return {
            "likely_winner": likely_winner,
            "confidence": confidence,
            "projected_final_scores": {
                "agent1": projected_agent1 if 'projected_agent1' in locals() else agent1_state.adjusted_total_payoff,
                "agent2": projected_agent2 if 'projected_agent2' in locals() else agent2_state.adjusted_total_payoff
            },
            "rounds_remaining": rounds_remaining
        }
    
    @staticmethod
    def _track_loss_aversion_effects(game_state: GameState) -> Dict[str, Any]:
        """Track real-time loss aversion effects"""
        game_history = game_state["game_history"]
        
        if not game_history:
            return {"no_effects_yet": True}
        
        recent_sucker_payoffs = []
        
        # Look at recent rounds for sucker payoffs
        for i, result in enumerate(game_history[-5:], start=len(game_history)-4):
            if result.agent1_move == Move.COOPERATE and result.agent2_move == Move.DEFECT:
                recent_sucker_payoffs.append({
                    "round": i,
                    "victim": "agent1",
                    "raw_payoff": result.agent1_payoff,
                    "adjusted_payoff": result.agent1_adjusted_payoff,
                    "penalty": result.agent1_payoff - result.agent1_adjusted_payoff
                })
            elif result.agent1_move == Move.DEFECT and result.agent2_move == Move.COOPERATE:
                recent_sucker_payoffs.append({
                    "round": i,
                    "victim": "agent2",
                    "raw_payoff": result.agent2_payoff,
                    "adjusted_payoff": result.agent2_adjusted_payoff,
                    "penalty": result.agent2_payoff - result.agent2_adjusted_payoff
                })
        
        return {
            "recent_sucker_payoffs": recent_sucker_payoffs,
            "loss_aversion_active": len(recent_sucker_payoffs) > 0,
            "total_penalty_recent": sum(sp["penalty"] for sp in recent_sucker_payoffs)
        }