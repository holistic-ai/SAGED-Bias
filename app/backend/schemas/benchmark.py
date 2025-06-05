from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class BenchmarkBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Benchmark name")
    description: Optional[str] = Field(None, description="Benchmark description")
    domain: str = Field(..., min_length=1, max_length=100, description="Domain category")
    categories: List[str] = Field(..., min_length=1, description="List of bias categories to test")
    data_tier: str = Field(default='split_sentences', description="Data processing tier")


class BenchmarkCreate(BenchmarkBase):
    config: Dict[str, Any] = Field(..., description="SAGED pipeline configuration")
    created_by: Optional[str] = Field(None, description="Creator username")


class BenchmarkUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    categories: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(draft|ready|processing|complete)$")
    is_active: Optional[bool] = None


class BenchmarkResponse(BenchmarkBase):
    id: int
    config: Dict[str, Any]
    status: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    data_file_path: Optional[str] = None
    source_file_path: Optional[str] = None

    class Config:
        from_attributes = True


class BenchmarkList(BaseModel):
    benchmarks: List[BenchmarkResponse]
    total: int
    page: int
    size: int


class BenchmarkStats(BaseModel):
    total_benchmarks: int
    by_domain: Dict[str, int]
    by_status: Dict[str, int]
    recent_activity: List[BenchmarkResponse] 