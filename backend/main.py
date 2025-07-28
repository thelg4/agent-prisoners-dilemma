"""
Loss-Averse Prisoner's Dilemma - Main FastAPI Application
Entry point for the backend server
"""

import uvicorn
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .api.routes import app as api_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Loss-Averse Prisoner's Dilemma API server...")
    logger.info("Server ready to accept connections")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
    # Clean up any resources here if needed


# Use the existing app from routes.py but add lifespan
app = api_app
app.router.lifespan_context = lifespan


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )