from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from .rational_agent import RationalAgent
from .loss_averse_agent import LossAverseAgent

class AgentFactory:
    """Factory for creating different types of agents"""
    
    @staticmethod
    def create_agent(
        agent_type: str, 
        agent_id: str, 
        llm_client: Any, 
        bias_config: Optional[Dict] = None
    ) -> BaseAgent:
        """Create an agent of the specified type"""
        
        if agent_type.lower() == "rational":
            return RationalAgent(agent_id, llm_client, bias_config)
        elif agent_type.lower() == "loss_averse":
            return LossAverseAgent(agent_id, llm_client, bias_config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    def get_available_types() -> List[str]:
        """Get list of available agent types"""
        return ["rational", "loss_averse"]
