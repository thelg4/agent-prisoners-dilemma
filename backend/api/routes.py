# backend/api/routes.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime
import logging

from ..game.models import GameState, AgentState, AgentType, GameConfig
from ..game.graph import PrisonersDilemmaGraph
from ..game.payoffs import PayoffMatrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loss-Averse Prisoner's Dilemma API",
    description="A behavioral economics twist on the classic game theory experiment",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game instances storage (use Redis/Database in production)
active_games: Dict[str, Dict[str, Any]] = {}


# Pydantic models for API
class GameConfigRequest(BaseModel):
    max_rounds: int = Field(default=10, ge=1, le=100, description="Number of rounds to play")
    loss_aversion_factor: float = Field(default=2.0, ge=1.0, le=5.0, description="Loss aversion multiplier")
    agent1_type: AgentType = Field(default=AgentType.ECON, description="Type of agent 1")
    agent2_type: AgentType = Field(default=AgentType.HUMAN, description="Type of agent 2")
    agent1_name: str = Field(default="Econ Agent", description="Name for agent 1")
    agent2_name: str = Field(default="Human Agent", description="Name for agent 2")
    reference_point: float = Field(default=2.5, description="Reference point for loss aversion")


class GameResponse(BaseModel):
    game_id: str
    status: str
    current_round: int
    max_rounds: int
    game_complete: bool
    winner: Optional[str] = None
    game_summary: Optional[Dict[str, Any]] = None
    agents: Dict[str, Dict[str, Any]]
    latest_round: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


class GameListResponse(BaseModel):
    games: List[Dict[str, Any]]
    total_count: int


class GameHistoryResponse(BaseModel):
    game_id: str
    history: List[Dict[str, Any]]
    total_rounds: int


class StepGameRequest(BaseModel):
    rounds: int = Field(default=1, ge=1, le=10, description="Number of rounds to advance")


@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "message": "Loss-Averse Prisoner's Dilemma API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/payoff-matrix", summary="Get Payoff Matrix")
async def get_payoff_matrix():
    """Get the prisoner's dilemma payoff matrix and explanations"""
    return {
        "matrix": {
            "cooperate_cooperate": PayoffMatrix.MATRIX[(PayoffMatrix.MATRIX.keys().__iter__().__next__())],
            "cooperate_defect": PayoffMatrix.MATRIX[list(PayoffMatrix.MATRIX.keys())[1]],
            "defect_cooperate": PayoffMatrix.MATRIX[list(PayoffMatrix.MATRIX.keys())[2]],
            "defect_defect": PayoffMatrix.MATRIX[list(PayoffMatrix.MATRIX.keys())[3]]
        },
        "outcomes": PayoffMatrix.get_all_outcomes(),
        "reference_point": PayoffMatrix.REFERENCE_POINT,
        "explanation": {
            "cooperate_cooperate": "Both agents cooperate - mutual reward",
            "cooperate_defect": "You cooperate, opponent defects - sucker's payoff",
            "defect_cooperate": "You defect, opponent cooperates - temptation payoff",
            "defect_defect": "Both agents defect - mutual punishment"
        }
    }


@app.post("/api/games", response_model=GameResponse, summary="Create New Game")
async def create_game(config: GameConfigRequest):
    """Create a new prisoner's dilemma game instance"""
    game_id = str(uuid.uuid4())
    
    logger.info(f"Creating new game {game_id} with config: {config}")
    
    # Create initial game state
    agent1_id = str(uuid.uuid4())
    agent2_id = str(uuid.uuid4())
    
    current_time = datetime.now()
    
    initial_state: GameState = {
        "game_id": game_id,
        "max_rounds": config.max_rounds,
        "current_round": 1,
        "loss_aversion_factor": config.loss_aversion_factor,
        "reference_point": config.reference_point,
        "agent1": AgentState(
            agent_id=agent1_id,
            agent_type=config.agent1_type,
            name=config.agent1_name
        ),
        "agent2": AgentState(
            agent_id=agent2_id,
            agent_type=config.agent2_type,
            name=config.agent2_name
        ),
        "game_history": [],
        "current_moves": {},
        "game_complete": False,
        "winner": None,
        "game_summary": None,
        "created_at": current_time,
        "updated_at": current_time
    }
    
    # Store the game state and graph instance
    game_graph = PrisonersDilemmaGraph()
    active_games[game_id] = {
        "state": initial_state,
        "graph": game_graph
    }
    
    logger.info(f"Game {game_id} created successfully")
    return _format_game_response(initial_state)


@app.post("/api/games/{game_id}/run", response_model=GameResponse, summary="Run Complete Game")
async def run_game(game_id: str, background_tasks: BackgroundTasks):
    """Run the complete game simulation"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game_data = active_games[game_id]
    state = game_data["state"]
    graph = game_data["graph"]
    
    if state["game_complete"]:
        raise HTTPException(status_code=400, detail="Game is already complete")
    
    logger.info(f"Running complete game {game_id}")
    
    try:
        # Run the complete game through LangGraph
        final_state = graph.graph.invoke(state)
        
        # Update stored state
        active_games[game_id]["state"] = final_state
        
        logger.info(f"Game {game_id} completed successfully")
        return _format_game_response(final_state)
    
    except Exception as e:
        logger.error(f"Game execution failed for {game_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Game execution failed: {str(e)}")


@app.post("/api/games/{game_id}/step", response_model=GameResponse, summary="Step Through Game")
async def step_game(game_id: str, step_request: StepGameRequest):
    """Step through the game one or more rounds at a time"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game_data = active_games[game_id]
    state = game_data["state"]
    graph = game_data["graph"]
    
    if state["game_complete"]:
        raise HTTPException(status_code=400, detail="Game is already complete")
    
    logger.info(f"Stepping game {game_id} for {step_request.rounds} rounds")
    
    try:
        # Run specified number of rounds
        for _ in range(step_request.rounds):
            if state["game_complete"]:
                break
            
            # Run one round through the graph
            # Note: This is a simplified approach. In production, you might want
            # to implement a more granular stepping mechanism
            state = graph.get_agent_moves(state)
            state = graph.calculate_payoffs(state)
            state = graph.update_agent_states(state)
            state = graph.check_game_end(state)
        
        # If game is complete, finalize it
        if state["game_complete"] and not state["game_summary"]:
            state = graph.finalize_game(state)
        
        # Update stored state
        active_games[game_id]["state"] = state
        
        return _format_game_response(state)
    
    except Exception as e:
        logger.error(f"Game stepping failed for {game_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Game stepping failed: {str(e)}")


@app.get("/api/games/{game_id}", response_model=GameResponse, summary="Get Game State")
async def get_game(game_id: str):
    """Get current game state"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    state = active_games[game_id]["state"]
    return _format_game_response(state)


@app.get("/api/games/{game_id}/history", response_model=GameHistoryResponse, summary="Get Game History")
async def get_game_history(game_id: str):
    """Get detailed game history"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    state = active_games[game_id]["state"]
    
    # Convert game results to dictionaries
    history = []
    for result in state["game_history"]:
        history.append({
            "round_number": result.round_number,
            "agent1_move": result.agent1_move.value,
            "agent2_move": result.agent2_move.value,
            "agent1_payoff": result.agent1_payoff,
            "agent2_payoff": result.agent2_payoff,
            "agent1_adjusted_payoff": result.agent1_adjusted_payoff,
            "agent2_adjusted_payoff": result.agent2_adjusted_payoff,
            "timestamp": result.timestamp.isoformat()
        })
    
    return GameHistoryResponse(
        game_id=game_id,
        history=history,
        total_rounds=len(history)
    )


@app.delete("/api/games/{game_id}", summary="Delete Game")
async def delete_game(game_id: str):
    """Delete a game instance"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    del active_games[game_id]
    logger.info(f"Game {game_id} deleted")
    return {"message": "Game deleted successfully"}


@app.get("/api/games", response_model=GameListResponse, summary="List All Games")
async def list_games(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None
):
    """List all active games with optional filtering"""
    games = []
    
    for game_id, game_data in active_games.items():
        state = game_data["state"]
        game_status = "complete" if state["game_complete"] else "active"
        
        # Apply status filter if provided
        if status and game_status != status:
            continue
        
        games.append({
            "game_id": game_id,
            "status": game_status,
            "current_round": state["current_round"],
            "max_rounds": state["max_rounds"],
            "created_at": state["created_at"].isoformat(),
            "updated_at": state["updated_at"].isoformat(),
            "agents": {
                "agent1": {
                    "name": state["agent1"].name,
                    "type": state["agent1"].agent_type.value
                },
                "agent2": {
                    "name": state["agent2"].name,
                    "type": state["agent2"].agent_type.value
                }
            },
            "winner": state["winner"],
            "loss_aversion_factor": state["loss_aversion_factor"]
        })
    
    # Apply pagination
    total_count = len(games)
    games = games[offset:offset + limit]
    
    return GameListResponse(games=games, total_count=total_count)


@app.post("/api/games/{game_id}/reset", response_model=GameResponse, summary="Reset Game")
async def reset_game(game_id: str):
    """Reset a game to its initial state"""
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game_data = active_games[game_id]
    state = game_data["state"]
    
    # Reset game state
    current_time = datetime.now()
    state["current_round"] = 1
    state["game_history"] = []
    state["current_moves"] = {}
    state["game_complete"] = False
    state["winner"] = None
    state["game_summary"] = None
    state["updated_at"] = current_time
    
    # Reset agent states
    state["agent1"].total_payoff = 0.0
    state["agent1"].adjusted_total_payoff = 0.0
    state["agent1"].moves_history = []
    state["agent1"].opponent_moves_history = []
    state["agent1"].cooperation_rate = 0.0
    state["agent1"].defection_rate = 0.0
    state["agent1"].average_payoff_per_round = 0.0
    
    state["agent2"].total_payoff = 0.0
    state["agent2"].adjusted_total_payoff = 0.0
    state["agent2"].moves_history = []
    state["agent2"].opponent_moves_history = []
    state["agent2"].cooperation_rate = 0.0
    state["agent2"].defection_rate = 0.0
    state["agent2"].average_payoff_per_round = 0.0
    
    # Create new graph instance
    game_data["graph"] = PrisonersDilemmaGraph()
    
    logger.info(f"Game {game_id} reset successfully")
    return _format_game_response(state)


def _format_game_response(state: GameState) -> GameResponse:
    """Format game state into API response"""
    latest_round = None
    if state["game_history"]:
        latest_result = state["game_history"][-1]
        latest_round = {
            "round_number": latest_result.round_number,
            "agent1_move": latest_result.agent1_move.value,
            "agent2_move": latest_result.agent2_move.value,
            "agent1_payoff": latest_result.agent1_payoff,
            "agent2_payoff": latest_result.agent2_payoff,
            "agent1_adjusted_payoff": latest_result.agent1_adjusted_payoff,
            "agent2_adjusted_payoff": latest_result.agent2_adjusted_payoff,
            "timestamp": latest_result.timestamp.isoformat()
        }
    
    return GameResponse(
        game_id=state["game_id"],
        status="complete" if state["game_complete"] else "active",
        current_round=state["current_round"],
        max_rounds=state["max_rounds"],
        game_complete=state["game_complete"],
        winner=state["winner"],
        game_summary=state["game_summary"],
        agents={
            "agent1": {
                "id": state["agent1"].agent_id,
                "name": state["agent1"].name,
                "type": state["agent1"].agent_type.value,
                "total_payoff": state["agent1"].total_payoff,
                "adjusted_total_payoff": state["agent1"].adjusted_total_payoff,
                "cooperation_rate": state["agent1"].cooperation_rate,
                "defection_rate": state["agent1"].defection_rate,
                "average_payoff": state["agent1"].average_payoff_per_round
            },
            "agent2": {
                "id": state["agent2"].agent_id,
                "name": state["agent2"].name,
                "type": state["agent2"].agent_type.value,
                "total_payoff": state["agent2"].total_payoff,
                "adjusted_total_payoff": state["agent2"].adjusted_total_payoff,
                "cooperation_rate": state["agent2"].cooperation_rate,
                "defection_rate": state["agent2"].defection_rate,
                "average_payoff": state["agent2"].average_payoff_per_round
            }
        },
        latest_round=latest_round,
        created_at=state["created_at"],
        updated_at=state["updated_at"]
    )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Resource not found", "status_code": 404}


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "status_code": 500}