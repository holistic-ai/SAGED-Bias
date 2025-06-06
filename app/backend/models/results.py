from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

# Import with fallback for direct execution
try:
    from ..database import Base
except ImportError:
    from database import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    
    # Reference to experiment
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    experiment = relationship("Experiment", back_populates="analysis_results")
    
    # Analysis metadata
    feature_name = Column(String(100), nullable=False, index=True)  # e.g., 'sentiment_score'
    analysis_type = Column(String(100), nullable=False, index=True)  # e.g., 'mean', 'disparity'
    target_group = Column(String(100), nullable=True, index=True)   # e.g., 'nationality', 'gender'
    baseline_group = Column(String(100), nullable=True)             # Reference group for comparison
    
    # Statistical results
    value = Column(Float, nullable=True)                # Main result value
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    effect_size = Column(Float, nullable=True)
    sample_size = Column(Integer, nullable=True)
    
    # Detailed results
    detailed_results = Column(JSON, nullable=True)     # Full analysis output
    visualization_data = Column(JSON, nullable=True)   # Data for charts/plots
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, feature='{self.feature_name}', type='{self.analysis_type}')>" 