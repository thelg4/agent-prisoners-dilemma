from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

@dataclass
class BiasConfig:
    """Configuration for cognitive biases"""
    loss_coefficient: float = 2.25
    gain_exponent: float = 0.88
    loss_exponent: float = 0.88
    reference_point_method: str = "moving_average"  # "moving_average", "fixed", "adaptive"
    reference_window_size: int = 10

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    agent_type: str
    agent_id: str
    bias_config: Optional[BiasConfig] = None
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    memory_window: int = 10

@dataclass
class TournamentConfig:
    """Configuration for tournaments"""
    num_rounds: int = 1000
    payoff_matrix: Dict[str, Any] = field(default_factory=lambda: {
        ("cooperate", "cooperate"): (3, 3),
        ("cooperate", "defect"): (0, 5),
        ("defect", "cooperate"): (5, 0),
        ("defect", "defect"): (1, 1)
    })
    tournament_type: str = "round_robin"

@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis"""
    alpha: float = 0.05
    multiple_comparison_correction: str = "bonferroni"
    effect_size_threshold: float = 0.3
    power_threshold: float = 0.8

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_id: str
    random_seed: int = 42
    num_replications: int = 10
    agent_configs: List[AgentConfig] = field(default_factory=list)
    tournament_config: TournamentConfig = field(default_factory=TournamentConfig)
    statistical_config: StatisticalConfig = field(default_factory=StatisticalConfig)
    output_directory: str = "results"
    save_raw_data: bool = True
    save_visualizations: bool = True

class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        
        return {
            "experiment": {
                "random_seed": 42,
                "num_replications": 10,
                "output_directory": "results"
            },
            "agents": {
                "rational": {
                    "llm_model": "gpt-4",
                    "llm_temperature": 0.1,
                    "memory_window": 10
                },
                "loss_averse": {
                    "llm_model": "gpt-4", 
                    "llm_temperature": 0.1,
                    "memory_window": 10,
                    "bias_config": {
                        "loss_coefficient": 2.25,
                        "gain_exponent": 0.88,
                        "loss_exponent": 0.88
                    }
                }
            },
            "tournament": {
                "num_rounds": 1000,
                "payoff_matrix": {
                    "cooperate_cooperate": [3, 3],
                    "cooperate_defect": [0, 5],
                    "defect_cooperate": [5, 0],
                    "defect_defect": [1, 1]
                }
            },
            "statistical": {
                "alpha": 0.05,
                "multiple_comparison_correction": "bonferroni",
                "effect_size_threshold": 0.3
            },
            "logging": {
                "level": "INFO",
                "save_logs": True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, file_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        if file_path is None:
            file_path = self.config_file or "config.json"
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def create_experiment_config(self, experiment_id: str) -> ExperimentConfig:
        """Create ExperimentConfig from current settings"""
        
        # Create agent configurations
        agent_configs = []
        
        # Rational agent
        agent_configs.append(AgentConfig(
            agent_type="rational",
            agent_id=f"rational_{experiment_id}",
            llm_model=self.get("agents.rational.llm_model", "gpt-4"),
            llm_temperature=self.get("agents.rational.llm_temperature", 0.1),
            memory_window=self.get("agents.rational.memory_window", 10)
        ))
        
        # Loss averse agent
        bias_config = BiasConfig(
            loss_coefficient=self.get("agents.loss_averse.bias_config.loss_coefficient", 2.25),
            gain_exponent=self.get("agents.loss_averse.bias_config.gain_exponent", 0.88),
            loss_exponent=self.get("agents.loss_averse.bias_config.loss_exponent", 0.88)
        )
        
        agent_configs.append(AgentConfig(
            agent_type="loss_averse",
            agent_id=f"loss_averse_{experiment_id}",
            bias_config=bias_config,
            llm_model=self.get("agents.loss_averse.llm_model", "gpt-4"),
            llm_temperature=self.get("agents.loss_averse.llm_temperature", 0.1),
            memory_window=self.get("agents.loss_averse.memory_window", 10)
        ))
        
        # Tournament configuration
        tournament_config = TournamentConfig(
            num_rounds=self.get("tournament.num_rounds", 1000)
        )
        
        # Statistical configuration
        statistical_config = StatisticalConfig(
            alpha=self.get("statistical.alpha", 0.05),
            multiple_comparison_correction=self.get("statistical.multiple_comparison_correction", "bonferroni"),
            effect_size_threshold=self.get("statistical.effect_size_threshold", 0.3)
        )
        
        return ExperimentConfig(
            experiment_id=experiment_id,
            random_seed=self.get("experiment.random_seed", 42),
            num_replications=self.get("experiment.num_replications", 10),
            agent_configs=agent_configs,
            tournament_config=tournament_config,
            statistical_config=statistical_config,
            output_directory=self.get("experiment.output_directory", "results")
        )