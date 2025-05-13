import os

# Project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Current directory (where this file is located)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
XNATION_DATA_DIR = os.path.join(DATA_DIR, "xntion")

# Xntion specific directories
XNATION_KEYWORDS_DIR = os.path.join(XNATION_DATA_DIR, "keywords")
XNATION_SOURCES_DIR = os.path.join(XNATION_DATA_DIR, "sources")
XNATION_SCRAPED_DIR = os.path.join(XNATION_DATA_DIR, "scraped")
XNATION_BENCHMARK_DIR = os.path.join(XNATION_DATA_DIR, "benchmark")
XNATION_FINAL_BENCHMARK_DIR = os.path.join(XNATION_DATA_DIR, "final_benchmark")

# File paths
XNATION_TEXT_PATH = os.path.join(CURRENT_DIR, "xnation.txt")  # Point to the actual text file
XNATION_REPLACEMENT_DESCRIPTION_PATH = os.path.join(XNATION_DATA_DIR, "replacement_description.json")
XNATION_GENERATIONS_PATH = os.path.join(XNATION_DATA_DIR, "generations.csv")
XNATION_EXTRACTIONS_PATH = os.path.join(XNATION_DATA_DIR, "extractions.csv")
XNATION_STATISTICS_PATH = os.path.join(XNATION_DATA_DIR, "statistics.csv")
XNATION_DISPARITY_PATH = os.path.join(XNATION_DATA_DIR, "disparity.csv")

# Directory containing LLMFactory.py and related files
CURRENT_DIR = os.path.join(PROJECT_ROOT, "xnation") 