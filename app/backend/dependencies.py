from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from .database import get_db
from .services import SAGEDService, ExperimentService
import sys
import os

# Add the project root to sys.path to import saged
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def get_database_session(db: Session = Depends(get_db)):
    """Dependency to get database session"""
    return db


# Global service instances
_saged_service = None
_experiment_service = None


def get_saged_service():
    """Dependency to get SAGED service instance"""
    global _saged_service
    if _saged_service is None:
        _saged_service = SAGEDService()
    return _saged_service


def get_experiment_service():
    """Dependency to get experiment service instance"""
    global _experiment_service
    if _experiment_service is None:
        _experiment_service = ExperimentService()
    return _experiment_service


def validate_experiment_config(config: dict):
    """Validate experiment configuration"""
    required_fields = ['categories', 'shared_config']
    
    for field in required_fields:
        if field not in config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required field: {field}"
            )
    
    return config 