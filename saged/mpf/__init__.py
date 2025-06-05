"""
MPF (Multi-Perspective Fusion) extension for SAGED.
Advanced bias mitigation using Multi-Perspective Fusion methodology.
"""

from .constantpath import (
    PROJECT_ROOT,
    CURRENT_DIR,
    DATA_DIR,
    MPF_DATA_DIR,
    MPF_KEYWORDS_DIR,
    MPF_SOURCES_DIR,
    MPF_SCRAPED_DIR,
    MPF_BENCHMARK_DIR,
    MPF_FINAL_BENCHMARK_DIR,
    MPF_TEXT_PATH,
    MPF_REPLACEMENT_DESCRIPTION_PATH,
    MPF_GENERATIONS_PATH,
    MPF_EXTRACTIONS_PATH,
    MPF_STATISTICS_PATH,
    MPF_DISPARITY_PATH,
)

from ._mpf_pipeline import mpf_pipeline
from ._mitigator import Mitigator
from .LLMFactory import LLMFactory

__all__ = [
    # Path configuration
    'PROJECT_ROOT',
    'CURRENT_DIR',
    'DATA_DIR',
    'MPF_DATA_DIR',
    'MPF_KEYWORDS_DIR',
    'MPF_SOURCES_DIR',
    'MPF_SCRAPED_DIR',
    'MPF_BENCHMARK_DIR',
    'MPF_FINAL_BENCHMARK_DIR',
    'MPF_TEXT_PATH',
    'MPF_REPLACEMENT_DESCRIPTION_PATH',
    'MPF_GENERATIONS_PATH',
    'MPF_EXTRACTIONS_PATH',
    'MPF_STATISTICS_PATH',
    'MPF_DISPARITY_PATH',
    # Core MPF components
    'mpf_pipeline',
    'Mitigator',
    'LLMFactory',
] 