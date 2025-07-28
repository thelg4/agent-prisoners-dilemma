from .models import Move

class PayoffMatrix:
    """Standard Prisoner's Dilemma payoff matrix with loss aversion calculations"""
    
    # Format: (agent1_payoff, agent2_payoff)
    MATRIX = {
        (Move.COOPERATE, Move.COOPERATE): (3, 3),  # Mutual cooperation - "Reward"
        (Move.COOPERATE, Move.DEFECT): (0, 5),     # Sucker's payoff / Temptation
        (Move.DEFECT, Move.COOPERATE): (5, 0),     # Temptation / Sucker's payoff  
        (Move.DEFECT, Move.DEFECT): (1, 1)         # Mutual defection - "Punishment"
    }
    
    REFERENCE_POINT = 2.5  # Midpoint for loss aversion calculations
    
    @classmethod
    def get_payoffs(cls, move1: Move, move2: Move) -> tuple:
        """Get raw payoffs from the matrix"""
        return cls.MATRIX[(move1, move2)]
    
    @classmethod
    def apply_loss_aversion(cls, payoff: float, loss_aversion_factor: float, 
                          reference_point: float = None) -> float:
        """
        Apply loss aversion to a payoff.
        Losses are felt more strongly than equivalent gains.
        
        Args:
            payoff: The raw payoff value
            loss_aversion_factor: How much more strongly losses are felt (typically 2.0-2.5)
            reference_point: The neutral point (default: 2.5)
            
        Returns:
            Adjusted payoff accounting for loss aversion
        """
        if reference_point is None:
            reference_point = cls.REFERENCE_POINT
            
        if payoff < reference_point:
            # It's a loss - multiply the loss by the aversion factor
            loss = reference_point - payoff
            return reference_point - (loss * loss_aversion_factor)
        else:
            # It's a gain - no adjustment needed
            return payoff
    
    @classmethod
    def get_all_outcomes(cls):
        """Get all possible outcomes with descriptions"""
        return {
            "mutual_cooperation": {
                "moves": (Move.COOPERATE, Move.COOPERATE),
                "payoffs": cls.MATRIX[(Move.COOPERATE, Move.COOPERATE)],
                "description": "Both cooperate - mutual benefit"
            },
            "mutual_defection": {
                "moves": (Move.DEFECT, Move.DEFECT),
                "payoffs": cls.MATRIX[(Move.DEFECT, Move.DEFECT)],
                "description": "Both defect - mutual punishment"
            },
            "agent1_exploits": {
                "moves": (Move.DEFECT, Move.COOPERATE),
                "payoffs": cls.MATRIX[(Move.DEFECT, Move.COOPERATE)],
                "description": "Agent 1 defects, Agent 2 cooperates"
            },
            "agent2_exploits": {
                "moves": (Move.COOPERATE, Move.DEFECT),
                "payoffs": cls.MATRIX[(Move.COOPERATE, Move.DEFECT)],
                "description": "Agent 1 cooperates, Agent 2 defects"
            }
        }
    
    @classmethod
    def calculate_expected_payoff(cls, my_move: Move, opponent_cooperation_prob: float) -> float:
        """
        Calculate expected payoff given opponent's cooperation probability
        
        Args:
            my_move: The move being considered
            opponent_cooperation_prob: Probability opponent will cooperate (0-1)
            
        Returns:
            Expected payoff
        """
        p_coop = opponent_cooperation_prob
        p_defect = 1 - p_coop
        
        if my_move == Move.COOPERATE:
            payoff_vs_coop = cls.MATRIX[(Move.COOPERATE, Move.COOPERATE)][0]
            payoff_vs_defect = cls.MATRIX[(Move.COOPERATE, Move.DEFECT)][0]
        else:
            payoff_vs_coop = cls.MATRIX[(Move.DEFECT, Move.COOPERATE)][0]
            payoff_vs_defect = cls.MATRIX[(Move.DEFECT, Move.DEFECT)][0]
        
        return (p_coop * payoff_vs_coop) + (p_defect * payoff_vs_defect)