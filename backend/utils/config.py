"""
Configuration management for the Loss-Averse Prisoner's Dilemma backend
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///./prisoners_dilemma.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class RedisConfig:
    """Redis configuration for caching and session storage"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = None
    decode_responses: bool = True


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    cors_origins: list = None
    max_games_per_user: int = 10
    game_timeout_minutes: int = 60


@dataclass
class GameConfig:
    """Default game configuration"""
    default_max_rounds: int = 10
    max_allowed_rounds: int = 100
    min_loss_aversion_factor: float = 1.0
    max_loss_aversion_factor: float = 5.0
    default_loss_aversion_factor: float = 2.0
    default_reference_point: float = 2.5
    cleanup_completed_games_hours: int = 24


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = None
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    max_connections_per_game: int = 50
    heartbeat_interval: int = 30
    connection_timeout: int = 300


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.environment = Environment(os.getenv("ENVIRONMENT", "development"))
        
        # Load configurations
        self.database = self._load_database_config()
        self.redis = self._load_redis_config()
        self.api = self._load_api_config()
        self.game = self._load_game_config()
        self.logging = self._load_logging_config()
        self.websocket = self._load_websocket_config()
        
        # Environment-specific overrides
        self._apply_environment_overrides()
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment"""
        return DatabaseConfig(
            url=os.getenv("DATABASE_URL", "sqlite:///./prisoners_dilemma.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        )
    
    def _load_redis_config(self) -> RedisConfig:
        """Load Redis configuration from environment"""
        return RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=os.getenv("REDIS_DECODE_RESPONSES", "true").lower() == "true"
        )
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment"""
        cors_origins = os.getenv("CORS_ORIGINS")
        if cors_origins:
            cors_origins = [origin.strip() for origin in cors_origins.split(",")]
        else:
            cors_origins = ["*"]  # Default to allow all origins (dev only)
        
        return APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            reload=os.getenv("API_RELOAD", "false").lower() == "true",
            cors_origins=cors_origins,
            max_games_per_user=int(os.getenv("MAX_GAMES_PER_USER", "10")),
            game_timeout_minutes=int(os.getenv("GAME_TIMEOUT_MINUTES", "60"))
        )
    
    def _load_game_config(self) -> GameConfig:
        """Load game configuration from environment"""
        return GameConfig(
            default_max_rounds=int(os.getenv("DEFAULT_MAX_ROUNDS", "10")),
            max_allowed_rounds=int(os.getenv("MAX_ALLOWED_ROUNDS", "100")),
            min_loss_aversion_factor=float(os.getenv("MIN_LOSS_AVERSION_FACTOR", "1.0")),
            max_loss_aversion_factor=float(os.getenv("MAX_LOSS_AVERSION_FACTOR", "5.0")),
            default_loss_aversion_factor=float(os.getenv("DEFAULT_LOSS_AVERSION_FACTOR", "2.0")),
            default_reference_point=float(os.getenv("DEFAULT_REFERENCE_POINT", "2.5")),
            cleanup_completed_games_hours=int(os.getenv("CLEANUP_COMPLETED_GAMES_HOURS", "24"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment"""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size_mb=int(os.getenv("LOG_MAX_FILE_SIZE_MB", "10")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
    
    def _load_websocket_config(self) -> WebSocketConfig:
        """Load WebSocket configuration from environment"""
        return WebSocketConfig(
            max_connections_per_game=int(os.getenv("WS_MAX_CONNECTIONS_PER_GAME", "50")),
            heartbeat_interval=int(os.getenv("WS_HEARTBEAT_INTERVAL", "30")),
            connection_timeout=int(os.getenv("WS_CONNECTION_TIMEOUT", "300"))
        )
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.DEVELOPMENT:
            self.api.debug = True
            self.api.reload = True
            self.database.echo = True
            self.logging.level = "DEBUG"
        
        elif self.environment == Environment.TESTING:
            self.database.url = "sqlite:///:memory:"
            self.redis.db = 1  # Use different Redis DB for tests
            self.logging.level = "WARNING"
        
        elif self.environment == Environment.PRODUCTION:
            self.api.debug = False
            self.api.reload = False
            self.database.echo = False
            self.logging.level = "INFO"
            if self.api.cors_origins == ["*"]:
                self.api.cors_origins = []  # Don't allow all origins in production
    
    def get_database_url(self) -> str:
        """Get the database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get the Redis URL"""
        password_part = f":{self.redis.password}@" if self.redis.password else ""
        return f"redis://{password_part}{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        return {
            "environment": self.environment.value,
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
                "max_games_per_user": self.api.max_games_per_user,
                "game_timeout_minutes": self.api.game_timeout_minutes
            },
            "game": {
                "default_max_rounds": self.game.default_max_rounds,
                "max_allowed_rounds": self.game.max_allowed_rounds,
                "loss_aversion_range": [
                    self.game.min_loss_aversion_factor,
                    self.game.max_loss_aversion_factor
                ],
                "default_loss_aversion_factor": self.game.default_loss_aversion_factor,
                "default_reference_point": self.game.default_reference_point
            },
            "websocket": {
                "max_connections_per_game": self.websocket.max_connections_per_game,
                "heartbeat_interval": self.websocket.heartbeat_interval
            }
        }


# Global configuration instance
config = Config()


# Configuration validation
def validate_config():
    """Validate the current configuration"""
    errors = []
    
    # Validate game configuration
    if config.game.min_loss_aversion_factor >= config.game.max_loss_aversion_factor:
        errors.append("min_loss_aversion_factor must be less than max_loss_aversion_factor")
    
    if config.game.default_loss_aversion_factor < config.game.min_loss_aversion_factor:
        errors.append("default_loss_aversion_factor must be >= min_loss_aversion_factor")
    
    if config.game.default_loss_aversion_factor > config.game.max_loss_aversion_factor:
        errors.append("default_loss_aversion_factor must be <= max_loss_aversion_factor")
    
    if config.game.default_max_rounds > config.game.max_allowed_rounds:
        errors.append("default_max_rounds must be <= max_allowed_rounds")
    
    # Validate API configuration
    if config.api.port < 1 or config.api.port > 65535:
        errors.append("API port must be between 1 and 65535")
    
    # Validate Redis configuration
    if config.redis.port < 1 or config.redis.port > 65535:
        errors.append("Redis port must be between 1 and 65535")
    
    if config.redis.db < 0:
        errors.append("Redis DB must be >= 0")
    
    # Validate WebSocket configuration
    if config.websocket.max_connections_per_game < 1:
        errors.append("max_connections_per_game must be >= 1")
    
    if config.websocket.heartbeat_interval < 5:
        errors.append("heartbeat_interval must be >= 5 seconds")
    
    if config.websocket.connection_timeout < config.websocket.heartbeat_interval:
        errors.append("connection_timeout must be >= heartbeat_interval")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True


# Environment setup helpers
def setup_logging():
    """Setup logging based on configuration"""
    import logging
    import logging.handlers
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format
    )
    
    # Add file handler if specified
    if config.logging.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            config.logging.file_path,
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.logging.format))
        logging.getLogger().addHandler(file_handler)


def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment"""
    return {
        "environment": config.environment.value,
        "python_version": os.sys.version,
        "config_summary": config.to_dict()
    }