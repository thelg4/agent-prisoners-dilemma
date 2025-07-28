"""
WebSocket implementation for real-time game updates
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GameWebSocketManager:
    """Manages WebSocket connections for real-time game updates"""
    
    def __init__(self):
        # Store active connections by game_id
        self.game_connections: Dict[str, List[WebSocket]] = {}
        # Store connection metadata
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, game_id: str, client_id: str = None):
        """Accept a new WebSocket connection for a specific game"""
        await websocket.accept()
        
        if game_id not in self.game_connections:
            self.game_connections[game_id] = []
        
        self.game_connections[game_id].append(websocket)
        self.connection_info[websocket] = {
            "game_id": game_id,
            "client_id": client_id or f"client_{len(self.game_connections[game_id])}",
            "connected_at": datetime.now()
        }
        
        logger.info(f"WebSocket connected for game {game_id}, client {client_id}")
        
        # Send welcome message
        await self.send_to_connection(websocket, {
            "type": "connection_established",
            "game_id": game_id,
            "client_id": self.connection_info[websocket]["client_id"],
            "timestamp": datetime.now().isoformat()
        })
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.connection_info:
            game_id = self.connection_info[websocket]["game_id"]
            client_id = self.connection_info[websocket]["client_id"]
            
            # Remove from game connections
            if game_id in self.game_connections:
                self.game_connections[game_id].remove(websocket)
                
                # Clean up empty game connection lists
                if not self.game_connections[game_id]:
                    del self.game_connections[game_id]
            
            # Remove connection info
            del self.connection_info[websocket]
            
            logger.info(f"WebSocket disconnected for game {game_id}, client {client_id}")
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_game(self, game_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections watching a specific game"""
        if game_id not in self.game_connections:
            return
        
        # Add timestamp to message
        message["timestamp"] = datetime.now().isoformat()
        
        # Send to all connections for this game
        disconnected_connections = []
        
        for websocket in self.game_connections[game_id]:
            try:
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected_connections.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            self.disconnect(websocket)
    
    async def broadcast_game_update(self, game_id: str, game_state: Dict[str, Any]):
        """Broadcast game state update to all watchers"""
        message = {
            "type": "game_update",
            "game_id": game_id,
            "data": game_state
        }
        await self.broadcast_to_game(game_id, message)
    
    async def broadcast_round_complete(self, game_id: str, round_data: Dict[str, Any]):
        """Broadcast round completion to all watchers"""
        message = {
            "type": "round_complete",
            "game_id": game_id,
            "data": round_data
        }
        await self.broadcast_to_game(game_id, message)
    
    async def broadcast_game_complete(self, game_id: str, final_summary: Dict[str, Any]):
        """Broadcast game completion to all watchers"""
        message = {
            "type": "game_complete",
            "game_id": game_id,
            "data": final_summary
        }
        await self.broadcast_to_game(game_id, message)
    
    async def broadcast_error(self, game_id: str, error_message: str):
        """Broadcast error message to all watchers"""
        message = {
            "type": "error",
            "game_id": game_id,
            "error": error_message
        }
        await self.broadcast_to_game(game_id, message)
    
    def get_connection_count(self, game_id: str) -> int:
        """Get number of active connections for a game"""
        return len(self.game_connections.get(game_id, []))
    
    def get_all_connections_info(self) -> Dict[str, Any]:
        """Get information about all active connections"""
        info = {
            "total_connections": len(self.connection_info),
            "games_with_connections": len(self.game_connections),
            "connections_by_game": {}
        }
        
        for game_id, connections in self.game_connections.items():
            info["connections_by_game"][game_id] = {
                "connection_count": len(connections),
                "clients": [
                    self.connection_info[ws]["client_id"] 
                    for ws in connections 
                    if ws in self.connection_info
                ]
            }
        
        return info


# Global WebSocket manager instance
websocket_manager = GameWebSocketManager()


# WebSocket endpoints (to be added to the main FastAPI app)
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """Main WebSocket endpoint for game updates"""
    client_id = websocket.query_params.get("client_id")
    
    try:
        await websocket_manager.connect(websocket, game_id, client_id)
        
        while True:
            # Listen for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, game_id, message)
            except json.JSONDecodeError:
                await websocket_manager.send_to_connection(websocket, {
                    "type": "error",
                    "error": "Invalid JSON message"
                })
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for game {game_id}: {e}")
        websocket_manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, game_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages from clients"""
    message_type = message.get("type")
    
    if message_type == "ping":
        # Respond to ping with pong
        await websocket_manager.send_to_connection(websocket, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "subscribe_to_rounds":
        # Client wants to receive round-by-round updates
        await websocket_manager.send_to_connection(websocket, {
            "type": "subscription_confirmed",
            "subscription": "rounds",
            "game_id": game_id
        })
    
    elif message_type == "request_game_state":
        # Client requesting current game state
        # This would integrate with the game storage to fetch current state
        await websocket_manager.send_to_connection(websocket, {
            "type": "game_state_requested",
            "message": "Game state request received - implement integration with game storage"
        })
    
    else:
        await websocket_manager.send_to_connection(websocket, {
            "type": "error",
            "error": f"Unknown message type: {message_type}"
        })


# Integration functions for game events
async def notify_game_created(game_id: str, game_config: Dict[str, Any]):
    """Notify about new game creation"""
    await websocket_manager.broadcast_to_game(game_id, {
        "type": "game_created",
        "game_id": game_id,
        "config": game_config
    })


async def notify_game_started(game_id: str):
    """Notify about game start"""
    await websocket_manager.broadcast_to_game(game_id, {
        "type": "game_started",
        "game_id": game_id
    })


async def notify_round_started(game_id: str, round_number: int):
    """Notify about round start"""
    await websocket_manager.broadcast_to_game(game_id, {
        "type": "round_started",
        "game_id": game_id,
        "round_number": round_number
    })


async def notify_moves_made(game_id: str, round_number: int, moves: Dict[str, str]):
    """Notify about agent moves"""
    await websocket_manager.broadcast_to_game(game_id, {
        "type": "moves_made",
        "game_id": game_id,
        "round_number": round_number,
        "moves": moves
    })


async def notify_payoffs_calculated(game_id: str, round_number: int, payoffs: Dict[str, Any]):
    """Notify about calculated payoffs"""
    await websocket_manager.broadcast_to_game(game_id, {
        "type": "payoffs_calculated",
        "game_id": game_id,
        "round_number": round_number,
        "payoffs": payoffs
    })