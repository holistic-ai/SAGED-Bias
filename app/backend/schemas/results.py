from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AnalysisResultBase(BaseModel):
    feature_name: str = Field(..., description="Feature name (e.g., 'sentiment_score')")
    analysis_type: str = Field(..., description="Analysis type (e.g., 'mean', 'disparity')")
    target_group: Optional[str] = Field(None, description="Target group for analysis")
    baseline_group: Optional[str] = Field(None, description="Baseline group for comparison")


class AnalysisResultResponse(AnalysisResultBase):
    id: int
    experiment_id: int
    value: Optional[float] = None
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    sample_size: Optional[int] = None
    detailed_results: Optional[Dict[str, Any]] = None
    visualization_data: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AnalysisResultList(BaseModel):
    results: List[AnalysisResultResponse]
    total: int
    page: int
    size: int


class BiasAnalysisSummary(BaseModel):
    experiment_id: int
    experiment_name: str
    benchmark_name: str
    total_features: int
    features_with_bias: int
    bias_percentage: float
    significant_disparities: List[AnalysisResultResponse]
    feature_summary: Dict[str, Dict[str, Any]]  # feature_name -> {mean_bias, max_bias, etc}


class ComparisonAnalysis(BaseModel):
    feature_name: str
    baseline_experiment_id: int
    comparison_experiment_id: int
    baseline_results: List[AnalysisResultResponse]
    comparison_results: List[AnalysisResultResponse]
    improvement_metrics: Dict[str, float]  # e.g., {"bias_reduction": 0.25, "p_value_improvement": 0.1}


class VisualizationData(BaseModel):
    chart_type: str = Field(..., description="Type of visualization (bar, line, heatmap, etc.)")
    title: str
    data: Dict[str, Any] = Field(..., description="Chart data in format suitable for frontend")
    config: Optional[Dict[str, Any]] = Field(None, description="Chart configuration options") 