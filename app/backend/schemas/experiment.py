from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ExperimentBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    benchmark_id: int = Field(..., description="Associated benchmark ID")


class ExperimentCreate(ExperimentBase):
    generation_config: Dict[str, Any] = Field(..., description="LLM generation configuration")
    extraction_config: Dict[str, Any] = Field(..., description="Feature extraction configuration")
    analysis_config: Dict[str, Any] = Field(..., description="Analysis configuration")
    created_by: Optional[str] = Field(None, description="Creator username")


class ExperimentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    extraction_config: Optional[Dict[str, Any]] = None
    analysis_config: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(created|running|completed|failed)$")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    error_message: Optional[str] = None


class ExperimentResponse(ExperimentBase):
    id: int
    generation_config: Dict[str, Any]
    extraction_config: Dict[str, Any]
    analysis_config: Dict[str, Any]
    status: str
    progress: float
    error_message: Optional[str] = None
    total_samples: Optional[int] = None
    features_extracted: Optional[List[str]] = None
    disparity_metrics: Optional[List[str]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    generation_file_path: Optional[str] = None
    extraction_file_path: Optional[str] = None
    analysis_file_path: Optional[str] = None

    class Config:
        from_attributes = True


class ExperimentList(BaseModel):
    experiments: List[ExperimentResponse]
    total: int
    page: int
    size: int


class ExperimentStats(BaseModel):
    total_experiments: int
    by_status: Dict[str, int]
    recent_activity: List[ExperimentResponse]
    avg_duration_seconds: Optional[float] = None


class ExperimentProgress(BaseModel):
    experiment_id: int
    status: str
    progress: float
    current_stage: str
    message: Optional[str] = None
    estimated_remaining_seconds: Optional[float] = None 