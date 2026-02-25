"""
AGATN Configuration Manager
Centralized configuration with environment variable loading and validation
"""
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, using system environment variables")

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation"""
    project_id: str
    credentials_path: str
    collection_prefix: str = "agatn"
    
    def __post_init__(self):
        if not self.project_id:
            raise ValueError("Firebase project_id is required")
        if not Path(self.credentials_path).exists():
            raise FileNotFoundError(f"Firebase credentials not found at {self.credentials_path}")

@dataclass
class ModelConfig:
    """Model hyperparameters and configurations"""
    # GAN Configuration
    gan_latent_dim: int = 100
    gan_hidden_dim: int = 256
    gan_learning_rate: float = 0.0002
    gan_critic_iterations: int = 5
    
    # GNN Configuration
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_num_heads: int = 8
    gnn_dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    sequence_length: int = 50
    num_assets: int = 10
    
    def validate(self):
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")
        if self.num_assets <= 0:
            raise ValueError("Number of assets must be positive")

class AGATNConfig:
    """Main configuration manager for AGATN"""
    
    def __init__(self):
        self.logging_level = os.getenv("LOGGING_LEVEL", "INFO")
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.model_dir = Path(os.getenv("MODEL_DIR", "./models"))
        
        # Initialize directories
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Firebase configuration
        self.firebase = FirebaseConfig(
            project_id=os.getenv("FIREBASE_PROJECT_ID", ""),
            credentials_path=os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase_credentials.json"),
            collection_prefix=os.getenv("FIREBASE_COLLECTION_PREFIX", "agatn")
        )
        
        # Model configuration
        self.models = ModelConfig()
        self.models.validate()
        
        # Trading parameters
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", "10000.0"))
        self.risk_free_rate = float(os.getenv("RISK_FREE_RATE", "0.02"))
        self.max_drawdown_limit = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.2"))
        
        # API Keys (if needed)
        self.ccxt_exchange = os.getenv("CCXT_EXCHANGE", "binance")
        self.api_key = os.getenv("API_KEY", "")
        self.api_secret = os.getenv("API_SECRET", "")
        
        self._validate_api_keys()
    
    def _validate_api_keys(self):
        """Validate API keys if trading is enabled"""
        trading_enabled = os.getenv("ENABLE_TRADING", "false").lower() == "true"
        if trading_enabled and (not self.api_key or not self.api_secret):
            logging.warning("Trading enabled but API keys are missing. Trading will be disabled.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging"""
        return {
            "logging_level": self.logging_level,
            "data_dir": str(self.data_dir),
            "model_dir": str(self.model_dir),
            "firebase_project": self.firebase.project_id,
            "trading_enabled": os.getenv("ENABLE_TRADING", "false")
        }

# Global configuration instance
config = AGATNConfig()