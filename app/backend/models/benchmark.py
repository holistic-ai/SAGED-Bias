from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

# Import with fallback for direct execution
try:
    from ..database import Base
except ImportError:
    from database import Base


class Benchmark(Base):
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    domain = Column(String(100), nullable=False, index=True)
    categories = Column(JSON, nullable=False)  # List of categories like ['nationality', 'gender']
    config = Column(JSON, nullable=False)  # Full SAGED configuration
    data_tier = Column(String(50), nullable=False, default='split_sentences')
    status = Column(String(50), nullable=False, default='draft')  # draft, ready, processing, complete
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(100), nullable=True)  # Simple string for now
    
    # File paths for stored data
    data_file_path = Column(String(500), nullable=True)
    source_file_path = Column(String(500), nullable=True)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="benchmark")

    def __repr__(self):
        return f"<Benchmark(id={self.id}, name='{self.name}', domain='{self.domain}')>" 