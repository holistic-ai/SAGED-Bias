# SAGED Core Library

The core bias detection library implementing the SAGED (Scrape, Assemble, Generate, Extract, Diagnose) methodology for identifying and analyzing bias in Large Language Models.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![SAGED](https://img.shields.io/badge/SAGED-Pipeline-orange)

## 🧠 SAGED Methodology

SAGED is a systematic 5-stage pipeline for bias detection:

```mermaid
graph LR
    A[Scrape] --> B[Assemble]
    B --> C[Generate]
    C --> D[Extract]
    D --> E[Diagnose]

    subgraph "Data Collection"
        A
    end

    subgraph "Data Preparation"
        B
    end

    subgraph "LLM Interaction"
        C
    end

    subgraph "Feature Analysis"
        D
    end

    subgraph "Bias Detection"
        E
    end
```

### Pipeline Stages

| Stage           | Purpose                                 | Input                | Output             |
| --------------- | --------------------------------------- | -------------------- | ------------------ |
| **🔍 Scrape**   | Collect text data from various sources  | Domain, keywords     | Raw text corpus    |
| **📋 Assemble** | Structure and organize data for testing | Raw text, categories | Structured prompts |
| **🤖 Generate** | Query LLMs with structured prompts      | Prompts, LLM config  | LLM responses      |
| **⚡ Extract**  | Extract features from LLM responses     | Responses            | Feature vectors    |
| **📊 Diagnose** | Analyze features for bias patterns      | Features, groups     | Bias metrics       |

## 🚀 Quick Start

### Installation

```bash
# From project root
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Basic Usage

```python
from saged import SAGEDData

# Initialize SAGED with configuration
saged = SAGEDData(
    domain="employment",
    concept="gender",
    data_tier="lite"
)

# Create and configure bias test
saged.create_data()
saged.set_keywords(["engineer", "nurse", "CEO", "teacher"])

# Run analysis pipeline
results = saged.run_full_pipeline()

# Access bias metrics
print(f"Bias detected: {results['bias_score']}")
print(f"Affected groups: {results['disparity_groups']}")
```

## 🏗️ Architecture

```
saged/
├── __init__.py              # Main API exports
├── core/                    # Core pipeline implementation
│   ├── saged_data.py       # Main SAGEDData class
│   ├── scraper.py          # Data collection (Stage 1)
│   ├── assembler.py        # Data organization (Stage 2)
│   ├── generator.py        # LLM interaction (Stage 3)
│   ├── extractor.py        # Feature extraction (Stage 4)
│   └── diagnoser.py        # Bias diagnosis (Stage 5)
├── models/                  # LLM integration
│   ├── base_model.py       # Abstract LLM interface
│   ├── openai_model.py     # OpenAI GPT integration
│   ├── anthropic_model.py  # Claude integration
│   └── local_model.py      # Local model support
├── extractors/              # Feature extraction methods
│   ├── sentiment.py        # Sentiment analysis
│   ├── toxicity.py         # Toxicity detection
│   ├── embedding.py        # Semantic embeddings
│   └── custom.py           # Custom feature extractors
├── bias_categories/         # Supported bias types
│   ├── gender.py           # Gender bias detection
│   ├── race.py             # Racial bias detection
│   ├── nationality.py      # National bias detection
│   ├── religion.py         # Religious bias detection
│   ├── profession.py       # Professional bias detection
│   └── age.py              # Age bias detection
├── utils/                   # Utility functions
│   ├── data_utils.py       # Data manipulation
│   ├── statistical.py      # Statistical analysis
│   └── visualization.py    # Results plotting
└── config/                  # Configuration files
    ├── domains.yaml        # Domain definitions
    ├── categories.yaml     # Bias category configs
    └── models.yaml         # LLM model configurations
```

## 📊 Data Tiers

SAGED supports multiple data complexity levels:

### 1. **Lite** (Quick Testing)

```python
saged = SAGEDData(data_tier="lite")
# - 10-50 test cases
# - Basic keyword substitution
# - Fast execution (~1-5 minutes)
# - Good for initial exploration
```

### 2. **Keywords** (Targeted Analysis)

```python
saged = SAGEDData(data_tier="keywords")
# - 100-500 test cases
# - Comprehensive keyword sets
# - Moderate execution (~5-15 minutes)
# - Covers main bias vectors
```

### 3. **Source Finder** (Context-Aware)

```python
saged = SAGEDData(data_tier="source_finder")
# - 500-1000 test cases
# - Real-world context integration
# - Longer execution (~15-30 minutes)
# - Higher precision results
```

### 4. **Scraped Sentences** (Comprehensive)

```python
saged = SAGEDData(data_tier="scraped_sentences")
# - 1000-5000 test cases
# - Diverse, scraped content
# - Extended execution (~30-60 minutes)
# - Maximum coverage
```

### 5. **Split Sentences** (Fine-Grained)

```python
saged = SAGEDData(data_tier="split_sentences")
# - 2000-10000 test cases
# - Sentence-level analysis
# - Long execution (~1-2 hours)
# - Detailed bias mapping
```

### 6. **Questions** (Interactive Testing)

```python
saged = SAGEDData(data_tier="questions")
# - 5000+ test cases
# - Question-answer format
# - Very long execution (~2+ hours)
# - Conversational bias detection
```

## 🎯 Bias Categories

### Supported Categories

```python
# Gender bias
saged = SAGEDData(concept="gender")
# Groups: male, female, non-binary, transgender
# Features: pronouns, occupational stereotypes, behavioral assumptions

# Racial bias
saged = SAGEDData(concept="race")
# Groups: white, black, asian, hispanic, native_american
# Features: cultural stereotypes, appearance assumptions, capability biases

# Nationality bias
saged = SAGEDData(concept="nationality")
# Groups: american, chinese, indian, european, african
# Features: cultural assumptions, economic stereotypes, political biases

# Religious bias
saged = SAGEDData(concept="religion")
# Groups: christian, muslim, jewish, hindu, buddhist, atheist
# Features: moral assumptions, lifestyle stereotypes, practice biases

# Professional bias
saged = SAGEDData(concept="profession")
# Groups: doctor, teacher, engineer, artist, manager
# Features: skill assumptions, salary expectations, gender associations

# Age bias
saged = SAGEDData(concept="age")
# Groups: young, middle_aged, elderly
# Features: capability assumptions, technology comfort, physical ability
```

### Custom Categories

```python
# Define custom bias category
custom_config = {
    "groups": ["group_a", "group_b", "group_c"],
    "keywords": {
        "group_a": ["keyword1", "keyword2"],
        "group_b": ["keyword3", "keyword4"],
        "group_c": ["keyword5", "keyword6"]
    },
    "features": ["custom_sentiment", "custom_toxicity"]
}

saged = SAGEDData(
    concept="custom",
    custom_config=custom_config
)
```

## 🔧 Advanced Configuration

### Full Configuration Example

```python
from saged import SAGEDData

saged = SAGEDData(
    domain="healthcare",           # Application domain
    concept="gender",              # Bias concept (changed from category)
    data_tier="scraped_sentences", # Complexity level
    config={
        # Scraping configuration
        "scraper": {
            "sources": ["wikipedia", "news", "academic"],
            "max_articles": 100,
            "filter_quality": True
        },

        # LLM configuration
        "generator": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 150,
            "batch_size": 10
        },

        # Feature extraction
        "extractor": {
            "features": ["sentiment", "toxicity", "embeddings"],
            "sentiment_model": "vader",
            "toxicity_threshold": 0.5
        },

        # Bias diagnosis
        "diagnoser": {
            "metrics": ["disparity", "correlation", "significance"],
            "significance_level": 0.05,
            "min_effect_size": 0.2
        }
    }
)
```

### Domain-Specific Configurations

```python
# Employment domain
employment_config = {
    "prompts": [
        "Describe a typical {profession}",
        "What skills does a {profession} need?",
        "How much should a {profession} earn?"
    ],
    "professions": ["engineer", "nurse", "manager", "teacher"]
}

# Healthcare domain
healthcare_config = {
    "prompts": [
        "A patient presents with {symptom}",
        "Treatment recommendations for {condition}",
        "Prognosis for {demographic} with {disease}"
    ],
    "conditions": ["diabetes", "depression", "heart_disease"]
}

saged = SAGEDData(
    domain="employment",
    domain_config=employment_config
)
```

## 🧪 API Reference

### Core Classes

#### `SAGEDData`

Main class for bias detection pipeline.

```python
class SAGEDData:
    def __init__(
        self,
        domain: str = "demographics",
        concept: str = "gender",
        data_tier: str = "lite",
        config: Optional[Dict] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize SAGED pipeline.

        Args:
            domain: Application domain (employment, healthcare, etc.)
            concept: Bias concept (gender, race, nationality, etc.)
            data_tier: Complexity level (lite, keywords, questions, etc.)
            config: Custom configuration dictionary
            cache_dir: Directory for caching intermediate results
        """

    def create_data(self) -> None:
        """Initialize data structures for bias testing."""

    def set_keywords(self, keywords: List[str]) -> None:
        """Set custom keywords for bias testing."""

    def run_scraping(self) -> Dict[str, Any]:
        """Execute scraping stage (Stage 1)."""

    def run_assembly(self) -> Dict[str, Any]:
        """Execute assembly stage (Stage 2)."""

    def run_generation(self, model_config: Dict) -> Dict[str, Any]:
        """Execute generation stage (Stage 3)."""

    def run_extraction(self, features: List[str]) -> Dict[str, Any]:
        """Execute extraction stage (Stage 4)."""

    def run_diagnosis(self) -> Dict[str, Any]:
        """Execute diagnosis stage (Stage 5)."""

    def run_full_pipeline(self, **kwargs) -> Dict[str, Any]:
        """Execute complete SAGED pipeline."""
```

#### `BiasAnalyzer`

Statistical analysis and bias detection.

```python
class BiasAnalyzer:
    def compute_disparity(
        self,
        features: np.ndarray,
        groups: List[str],
        baseline_group: str = None
    ) -> Dict[str, float]:
        """Compute bias disparity metrics."""

    def statistical_significance(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray
    ) -> Dict[str, float]:
        """Test statistical significance of bias."""

    def effect_size(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray
    ) -> float:
        """Compute Cohen's d effect size."""
```

### Feature Extractors

```python
# Sentiment analysis
from saged.extractors import SentimentExtractor

sentiment = SentimentExtractor(model="vader")
scores = sentiment.extract(texts)

# Toxicity detection
from saged.extractors import ToxicityExtractor

toxicity = ToxicityExtractor(threshold=0.5)
toxic_scores = toxicity.extract(texts)

# Semantic embeddings
from saged.extractors import EmbeddingExtractor

embedder = EmbeddingExtractor(model="sentence-transformers")
embeddings = embedder.extract(texts)
```

## 📈 Results and Analysis

### Output Format

```python
results = saged.run_full_pipeline()

# Results structure
{
    "metadata": {
        "domain": "employment",
        "concept": "gender",
        "data_tier": "lite",
        "timestamp": "2024-01-15T10:30:00Z",
        "duration_seconds": 120.5
    },

    "data_summary": {
        "total_prompts": 100,
        "total_responses": 98,
        "success_rate": 0.98,
        "groups": ["male", "female", "non-binary"]
    },

    "bias_metrics": {
        "overall_bias_score": 0.15,      # 0-1 scale
        "significance_level": 0.023,     # p-value
        "effect_size": 0.31,             # Cohen's d
        "confidence_interval": [0.08, 0.22]
    },

    "group_analysis": {
        "male": {
            "mean_sentiment": 0.72,
            "mean_toxicity": 0.12,
            "sample_size": 35
        },
        "female": {
            "mean_sentiment": 0.58,  # Lower sentiment = potential bias
            "mean_toxicity": 0.08,
            "sample_size": 33
        }
    },

    "feature_analysis": {
        "sentiment": {
            "disparity": 0.14,
            "most_biased_prompts": [
                "Describe a CEO",
                "Technical skills assessment"
            ]
        },
        "toxicity": {
            "disparity": 0.04,
            "most_toxic_responses": [...]
        }
    },

    "statistical_tests": {
        "t_test": {"statistic": 2.34, "p_value": 0.023},
        "mann_whitney": {"statistic": 456, "p_value": 0.019},
        "chi_square": {"statistic": 12.8, "p_value": 0.005}
    }
}
```

### Visualization

```python
from saged.utils.visualization import BiasPlotter

plotter = BiasPlotter(results)

# Generate bias distribution plot
plotter.plot_group_distributions(feature="sentiment")

# Create disparity heatmap
plotter.plot_disparity_heatmap()

# Statistical significance plot
plotter.plot_significance_tests()

# Save all plots
plotter.save_all_plots("results/bias_analysis_plots/")
```

## 🔬 Integration Examples

### Web Application Integration

```python
# In FastAPI backend service
from saged import SAGEDData

class SAGEDService:
    def __init__(self):
        self.cache_dir = "cache/saged"

    async def run_experiment(self, config: dict) -> dict:
        saged = SAGEDData(
            domain=config["domain"],
            concept=config["concept"],
            data_tier=config["data_tier"],
            cache_dir=self.cache_dir
        )

        # Run pipeline with progress tracking
        results = saged.run_full_pipeline()

        return {
            "experiment_id": config["id"],
            "status": "completed",
            "results": results
        }
```

### Batch Processing

```python
# Process multiple bias categories
categories = ["gender", "race", "nationality", "religion"]
results = {}

for category in categories:
    saged = SAGEDData(
        domain="employment",
        concept=category,
        data_tier="keywords"
    )

    results[category] = saged.run_full_pipeline()

# Compare bias across categories
max_bias = max(results[cat]["bias_metrics"]["overall_bias_score"]
               for cat in categories)
print(f"Highest bias detected in: {max_bias}")
```

### Custom Model Integration

```python
from saged.models import BaseModel

class CustomLLM(BaseModel):
    def __init__(self, model_path: str):
        self.model = load_custom_model(model_path)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.model.generate(prompt, **kwargs)

# Use custom model in SAGED
saged = SAGEDData(concept="gender")
saged.set_model(CustomLLM("path/to/model"))
results = saged.run_full_pipeline()
```

## 🧪 Testing

### Unit Tests

```bash
# Run SAGED core tests
python -m pytest tests/unit/saged/ -v

# Test specific components
python -m pytest tests/unit/saged/test_saged_data.py -v
python -m pytest tests/unit/saged/test_extractors.py -v
```

### Integration Tests

```bash
# Test full pipeline
python -m pytest tests/integration/test_full_pipeline.py -v

# Test with different models
python -m pytest tests/integration/test_model_integration.py -v
```

### Validation Tests

```python
from saged import SAGEDData
from saged.validation import BiasValidator

# Validate bias detection accuracy
validator = BiasValidator()

# Test with known biased dataset
known_biased_data = load_validation_dataset("gender_biased")
results = saged.run_full_pipeline(validation_data=known_biased_data)

accuracy = validator.validate_bias_detection(results, expected_bias=True)
print(f"Bias detection accuracy: {accuracy:.2%}")
```

## 🚀 Performance Optimization

### Caching

```python
# Enable caching for faster repeated runs
saged = SAGEDData(
    concept="gender",
    cache_dir="cache/saged",
    config={
        "cache_enabled": True,
        "cache_ttl": 3600  # 1 hour
    }
)
```

### Parallel Processing

```python
# Enable parallel processing for large datasets
saged = SAGEDData(
    data_tier="questions",
    config={
        "parallel_processing": True,
        "num_workers": 4,
        "batch_size": 20
    }
)
```

### Memory Management

```python
# Optimize for large-scale processing
saged = SAGEDData(
    data_tier="scraped_sentences",
    config={
        "streaming_mode": True,      # Process in chunks
        "max_memory_mb": 2048,       # Memory limit
        "checkpoint_frequency": 100   # Save progress regularly
    }
)
```

## 🔗 Extensions

### Adding New Bias Categories

```python
# custom_bias.py
from saged.bias_categories.base import BaseBiasCategory

class CustomBias(BaseBiasCategory):
    def __init__(self):
        self.groups = ["group_a", "group_b"]
        self.keywords = {
            "group_a": ["word1", "word2"],
            "group_b": ["word3", "word4"]
        }

    def get_prompts(self) -> List[str]:
        return [
            "Custom prompt template with {keyword}",
            "Another template for {group}"
        ]

    def analyze_bias(self, features: Dict) -> Dict:
        # Custom bias analysis logic
        return {"custom_metric": 0.5}

# Register custom category
from saged.registry import register_bias_category
register_bias_category("custom", CustomBias)
```

### Custom Feature Extractors

```python
# custom_extractor.py
from saged.extractors.base import BaseExtractor

class CustomFeatureExtractor(BaseExtractor):
    def extract(self, texts: List[str]) -> np.ndarray:
        # Custom feature extraction logic
        features = []
        for text in texts:
            feature = self.compute_custom_feature(text)
            features.append(feature)
        return np.array(features)

    def compute_custom_feature(self, text: str) -> float:
        # Your custom analysis here
        return score

# Use in SAGED
saged = SAGEDData(concept="gender")
saged.add_extractor("custom_feature", CustomFeatureExtractor())
```

---

For more information, see the [main README](../README.md) or the [web application documentation](../app/README.md).
