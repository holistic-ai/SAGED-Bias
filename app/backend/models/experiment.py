from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

# Import with fallback for direct execution
try:
    from ..database import Base
except ImportError:
    from database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Reference to benchmark
    benchmark_id = Column(Integer, ForeignKey("benchmarks.id"), nullable=False)
    benchmark = relationship("Benchmark", back_populates="experiments")
    
    # Experiment configuration
    generation_config = Column(JSON, nullable=False)  # LLM generation settings
    extraction_config = Column(JSON, nullable=False)  # Feature extraction settings
    analysis_config = Column(JSON, nullable=False)    # Analysis settings
    
    # Execution status
    status = Column(String(50), nullable=False, default='created')  # created, running, completed, failed
    progress = Column(Float, default=0.0)  # 0.0 to 1.0
    error_message = Column(Text, nullable=True)
    
    # Results metadata
    total_samples = Column(Integer, nullable=True)
    features_extracted = Column(JSON, nullable=True)  # List of extracted features
    disparity_metrics = Column(JSON, nullable=True)   # List of computed metrics
    
    # Execution timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(100), nullable=True)
    
    # File paths for results
    generation_file_path = Column(String(500), nullable=True)
    extraction_file_path = Column(String(500), nullable=True)
    analysis_file_path = Column(String(500), nullable=True)
    
    # Relationships
    analysis_results = relationship("AnalysisResult", back_populates="experiment")

    def __repr__(self):
        return f"<Experiment(id={self.id}, name='{self.name}', status='{self.status}')>" 