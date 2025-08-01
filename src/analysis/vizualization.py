from typing import List, Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from ..game.tournament import TournamentResult
from .behavioral_metrics import BehavioralProfile, BehavioralMetrics

class Visualization:
    """Create visualizations for tournament analysis and articles"""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        plt.style.use(style)
        sns.set_palette("husl")
        self.behavioral_metrics = BehavioralMetrics()
    
    def plot_cooperation_evolution(
        self, 
        tournament_result: TournamentResult,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> go.Figure:
        """Plot cooperation rate evolution over time"""
        
        cooperation_by_round = tournament_result.cooperation_by_round
        
        # Calculate moving averages
        window_size = 50
        agent1_evolution = self.behavioral_metrics.calculate_cooperation_evolution(
            cooperation_by_round, window_size
        )
        agent2_evolution = self.behavioral_metrics.calculate_cooperation_evolution(
            [(coop[1], coop[0]) for coop in cooperation_by_round], window_size
        )
        
        rounds = list(range(1, len(cooperation_by_round) + 1))
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rounds,
            y=agent1_evolution,
            mode='lines',
            name=f'{tournament_result.agent1_id} ({tournament_result.agent1_type})',
            line=dict(width=3),
            hovertemplate='Round: %{x}<br>Cooperation Rate: %{y:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds,
            y=agent2_evolution,
            mode='lines',
            name=f'{tournament_result.agent2_id} ({tournament_result.agent2_type})',
            line=dict(width=3),
            hovertemplate='Round: %{x}<br>Cooperation Rate: %{y:.2%}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Cooperation Rate Evolution: {tournament_result.tournament_id}',
            xaxis_title='Round Number',
            yaxis_title='Cooperation Rate',
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_score_comparison(
        self,
        tournament_results: List[TournamentResult],
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> go.Figure:
        """Plot score comparison across multiple tournaments"""
        
        # Prepare data
        data = []
        for result in tournament_results:
            data.extend([
                {
                    'Agent': f"{result.agent1_id} ({result.agent1_type})",
                    'Score': result.agent1_final_score,
                    'Tournament': result.tournament_id,
                    'Agent_Type': result.agent1_type
                },
                {
                    'Agent': f"{result.agent2_id} ({result.agent2_type})",
                    'Score': result.agent2_final_score,
                    'Tournament': result.tournament_id,
                    'Agent_Type': result.agent2_type
                }
            ])
        
        df = pd.DataFrame(data)
        
        # Create box plot
        fig = px.box(
            df, 
            x='Agent_Type', 
            y='Score', 
            color='Agent_Type',
            title='Score Distribution by Agent Type',
            points='all'
        )
        
        fig.update_layout(
            template='plotly_white',
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_cooperation_heatmap(
        self,
        tournament_result: TournamentResult,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Create heatmap of cooperation patterns"""
        
        cooperation_by_round = tournament_result.cooperation_by_round
        
        # Reshape data into chunks for heatmap
        chunk_size = 50
        chunks = [cooperation_by_round[i:i+chunk_size] 
                 for i in range(0, len(cooperation_by_round), chunk_size)]
        
        # Calculate cooperation rates for each chunk
        heatmap_data = []
        for i, chunk in enumerate(chunks):
            agent1_coop_rate = sum(1 for coop in chunk if coop[0]) / len(chunk)
            agent2_coop_rate = sum(1 for coop in chunk if coop[1]) / len(chunk)
            heatmap_data.append([agent1_coop_rate, agent2_coop_rate])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        heatmap_array = np.array(heatmap_data).T
        
        im = ax.imshow(heatmap_array, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(range(len(chunks)))
        ax.set_xticklabels([f'Rounds {i*chunk_size+1}-{min((i+1)*chunk_size, len(cooperation_by_round))}' 
                           for i in range(len(chunks))], rotation=45)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([
            f'{tournament_result.agent1_id} ({tournament_result.agent1_type})',
            f'{tournament_result.agent2_id} ({tournament_result.agent2_type})'
        ])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cooperation Rate', rotation=270, labelpad=20)
        
        # Add title
        plt.title(f'Cooperation Heatmap: {tournament_result.tournament_id}', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_behavioral_profile_radar(
        self,
        behavioral_profiles: List[BehavioralProfile],
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> go.Figure:
        """Create radar chart of behavioral profiles"""
        
        # Define metrics for radar chart
        metrics = [
            'Cooperation Rate',
            'Consistency',
            'Adaptation Rate',
            'Score Performance',
            'Risk Tolerance'
        ]
        
        fig = go.Figure()
        
        for profile in behavioral_profiles:
            # Normalize metrics to 0-1 scale
            values = [
                profile.avg_cooperation_rate,
                profile.consistency_score,
                min(profile.adaptation_rate / 0.1, 1.0),  # Cap at 0.1 for normalization
                min(profile.avg_final_score / 3000, 1.0),  # Assume max score ~3000
                1.0 - profile.cooperation_variance  # Lower variance = higher tolerance
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=f'{profile.agent_id} ({profile.agent_type})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Behavioral Profile Comparison",
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        if show_plot:
            fig.show()
        
        return fig
    
    def create_article_dashboard(
        self,
        tournament_results: List[TournamentResult],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create comprehensive dashboard for Medium article"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Cooperation Rates Over Time',
                'Final Score Distribution', 
                'Head-to-Head Performance',
                'Strategy Evolution'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Plot 1: Cooperation evolution for first tournament
        if tournament_results:
            main_result = tournament_results[0]
            cooperation_by_round = main_result.cooperation_by_round
            
            agent1_evolution = self.behavioral_metrics.calculate_cooperation_evolution(
                cooperation_by_round, 50
            )
            agent2_evolution = self.behavioral_metrics.calculate_cooperation_evolution(
                [(coop[1], coop[0]) for coop in cooperation_by_round], 50
            )
            
            rounds = list(range(1, len(cooperation_by_round) + 1))
            
            fig.add_trace(
                go.Scatter(x=rounds, y=agent1_evolution, name=main_result.agent1_type,
                          line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=rounds, y=agent2_evolution, name=main_result.agent2_type,
                          line=dict(color='red')),
                row=1, col=1
            )
        
        # Plot 2: Score distribution
        scores_data = []
        for result in tournament_results:
            scores_data.extend([
                result.agent1_final_score, result.agent2_final_score
            ])
        
        fig.add_trace(
            go.Histogram(x=scores_data, name='Score Distribution', 
                        marker_color='lightblue'),
            row=1, col=2
        )
        
        # Plot 3: Head-to-head wins
        agent_types = set()
        for result in tournament_results:
            agent_types.add(result.agent1_type)
            agent_types.add(result.agent2_type)
        
        win_matrix = {agent_type: {other_type: 0 for other_type in agent_types} 
                     for agent_type in agent_types}
        
        for result in tournament_results:
            if result.agent1_final_score > result.agent2_final_score:
                win_matrix[result.agent1_type][result.agent2_type] += 1
            else:
                win_matrix[result.agent2_type][result.agent1_type] += 1
        
        # Convert to heatmap format
        agent_type_list = list(agent_types)
        z_data = [[win_matrix[row_type][col_type] for col_type in agent_type_list] 
                  for row_type in agent_type_list]
        
        fig.add_trace(
            go.Heatmap(z=z_data, x=agent_type_list, y=agent_type_list,
                      colorscale='Blues', name='Wins'),
            row=2, col=1
        )
        
        # Plot 4: Strategy evolution (cooperation rate changes)
        if tournament_results:
            evolution_data = []
            for result in tournament_results:
                cooperation_evolution = self.behavioral_metrics.calculate_cooperation_evolution(
                    result.cooperation_by_round
                )
                early_coop = np.mean(cooperation_evolution[:100]) if len(cooperation_evolution) >= 100 else cooperation_evolution[0]
                late_coop = np.mean(cooperation_evolution[-100:]) if len(cooperation_evolution) >= 100 else cooperation_evolution[-1]
                evolution_data.append({
                    'Agent_Type': result.agent1_type,
                    'Early_Cooperation': early_coop,
                    'Late_Cooperation': late_coop,
                    'Change': late_coop - early_coop
                })
            
            df_evolution = pd.DataFrame(evolution_data)
            
            fig.add_trace(
                go.Scatter(x=df_evolution['Early_Cooperation'], 
                          y=df_evolution['Late_Cooperation'],
                          mode='markers+text',
                          text=df_evolution['Agent_Type'],
                          name='Strategy Evolution'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Tournament Analysis Dashboard",
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
