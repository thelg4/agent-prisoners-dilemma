import logging
import logging.config
import os
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    experiment_id: Optional[str] = None
) -> None:
    """Set up logging configuration for the project"""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_id:
            log_file = f"{log_dir}/experiment_{experiment_id}_{timestamp}.log"
        else:
            log_file = f"{log_dir}/loss_averse_pd_{timestamp}.log"
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': log_level,
                'formatter': 'detailed',
                'filename': log_file,
                'mode': 'w'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': log_level,
                'handlers': ['console', 'file']
            }
        }
    }
    
    logging.config.dictConfig(config)
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")

class ExperimentLogger:
    """Specialized logger for experiment tracking"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.logger = logging.getLogger(f"experiment.{experiment_id}")
        self.start_time = datetime.now()
        
    def log_experiment_start(self, config: Dict[str, Any]) -> None:
        """Log experiment start with configuration"""
        self.logger.info(f"Starting experiment {self.experiment_id}")
        self.logger.info(f"Configuration: {config}")
        
    def log_tournament_start(self, tournament_id: str, agent1_id: str, agent2_id: str) -> None:
        """Log tournament start"""
        self.logger.info(f"Tournament {tournament_id}: {agent1_id} vs {agent2_id}")
        
    def log_tournament_result(self, tournament_result) -> None:
        """Log tournament completion"""
        self.logger.info(f"Tournament {tournament_result.tournament_id} completed")
        self.logger.info(f"Final scores: {tournament_result.agent1_id}={tournament_result.agent1_final_score}, "
                        f"{tournament_result.agent2_id}={tournament_result.agent2_final_score}")
        
    def log_statistical_result(self, test_name: str, result) -> None:
        """Log statistical test result"""
        self.logger.info(f"Statistical test {test_name}: {result.interpretation}")
        
    def log_experiment_end(self) -> None:
        """Log experiment completion"""
        duration = datetime.now() - self.start_time
        self.logger.info(f"Experiment {self.experiment_id} completed in {duration}")