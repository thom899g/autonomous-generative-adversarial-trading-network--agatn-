"""
Firebase Client for AGATN
Handles all Firebase interactions with robust error handling and connection management
"""
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
from pathlib import Path

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, initialize_app
    from firebase_admin.exceptions import FirebaseError
    FIREBASE_AVAILABLE = True
except ImportError:
    logging.error("firebase-admin not installed. Install with: pip install firebase-admin")
    FIREBASE_AVAILABLE = False

class FirebaseClient:
    """Firebase client with connection pooling and error recovery"""
    
    def __init__(self, config):
        self.config = config
        self._db = None
        self._app = None
        self._connected = False
        self.logger = logging.getLogger(__name__)
        self._initialize()
    
    def _initialize(self):
        """Initialize Firebase connection with error handling"""
        if not FIREBASE_AVAILABLE:
            self.logger.error("Firebase libraries not available")
            return
        
        try:
            cred_path = Path(self.config.firebase.credentials_path)
            if not cred_path.exists():
                raise FileNotFoundError(f"Credentials file not found: {cred_path}")
            
            cred = credentials.Certificate(str(cred_path))
            self._app = initialize_app(cred, {
                'projectId': self.config.firebase.project_id
            })
            self._db = firestore.client(app=self._app)
            self._connected = True
            self.logger.info(f"Firebase connected to project: {self.config.firebase.project_id}")
            
        except FirebaseError as e:
            self.logger.error(f"Firebase initialization error: {str(e)}")
            self._connected = False
        except Exception as e:
            self.logger.error(f"Unexpected error during Firebase init: {str(e)}")
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if Firebase connection is active"""
        return self._connected and self._db is not None
    
    def save_market_data(self, symbol: str, data: Dict[str, Any], timestamp: Optional[