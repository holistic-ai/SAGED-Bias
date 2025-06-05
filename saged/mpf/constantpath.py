import os

# Project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Current directory (where this file is located)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MPF_DATA_DIR = os.path.join(DATA_DIR, "mpf")

# MPF specific directories
MPF_KEYWORDS_DIR = os.path.join(MPF_DATA_DIR, "keywords")
MPF_SOURCES_DIR = os.path.join(MPF_DATA_DIR, "sources")
MPF_SCRAPED_DIR = os.path.join(MPF_DATA_DIR, "scraped")
MPF_BENCHMARK_DIR = os.path.join(MPF_DATA_DIR, "benchmark")
MPF_FINAL_BENCHMARK_DIR = os.path.join(MPF_DATA_DIR, "final_benchmark")

# File paths
MPF_TEXT_PATH = os.path.join(CURRENT_DIR, "xnation.txt")  # Point to the actual text file
MPF_REPLACEMENT_DESCRIPTION_PATH = os.path.join(MPF_DATA_DIR, "replacement_description.json")
MPF_GENERATIONS_PATH = os.path.join(MPF_DATA_DIR, "generations.csv")
MPF_EXTRACTIONS_PATH = os.path.join(MPF_DATA_DIR, "extractions.csv")
MPF_STATISTICS_PATH = os.path.join(MPF_DATA_DIR, "statistics.csv")
MPF_DISPARITY_PATH = os.path.join(MPF_DATA_DIR, "disparity.csv")

# Directory containing LLMFactory.py and related files
CURRENT_DIR = os.path.join(PROJECT_ROOT, "saged", "mpf") 