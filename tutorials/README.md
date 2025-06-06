# SAGED Tutorials & Examples

Interactive tutorials and examples for learning the SAGED bias detection methodology and platform usage.

![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![SAGED](https://img.shields.io/badge/SAGED-Tutorial-green)

## 🚀 Quick Start

### Setup Environment

```bash
# From project root
pip install jupyter notebook
pip install -r requirements.txt

# Start Jupyter
cd tutorials
jupyter notebook
```

### Open Your First Tutorial

```bash
# Start with the basics
open 01_introduction_to_saged.ipynb

# Or view online
jupyter notebook 01_introduction_to_saged.ipynb
```

## 📚 Tutorial Structure

```
tutorials/
├── beginner/               # Getting started tutorials
│   ├── 01_introduction_to_saged.ipynb
│   ├── 02_basic_bias_detection.ipynb
│   ├── 03_understanding_data_tiers.ipynb
│   └── 04_interpreting_results.ipynb
├── intermediate/           # Advanced usage
│   ├── 05_custom_bias_categories.ipynb
│   ├── 06_feature_extraction.ipynb
│   ├── 07_statistical_analysis.ipynb
│   └── 08_web_platform_usage.ipynb
├── advanced/               # Expert-level topics
│   ├── 09_custom_models.ipynb
│   ├── 10_pipeline_optimization.ipynb
│   ├── 11_batch_processing.ipynb
│   └── 12_research_applications.ipynb
├── examples/               # Real-world examples
│   ├── employment_bias_study.ipynb
│   ├── healthcare_bias_analysis.ipynb
│   ├── education_bias_detection.ipynb
│   └── social_media_bias_study.ipynb
├── datasets/               # Sample datasets
│   ├── employment_prompts.csv
│   ├── healthcare_scenarios.json
│   └── social_media_posts.csv
└── assets/                 # Images, diagrams, resources
    ├── saged_pipeline.png
    ├── bias_examples.png
    └── methodology_diagram.svg
```

## 🎓 Learning Paths

### 🌱 **Beginner Path** (2-3 hours)

**Perfect for**: New users, researchers, students

1. **[Introduction to SAGED](beginner/01_introduction_to_saged.ipynb)** (30 mins)

   - What is bias in AI?
   - SAGED methodology overview
   - Installation and setup

2. **[Basic Bias Detection](beginner/02_basic_bias_detection.ipynb)** (45 mins)

   - Your first bias analysis
   - Gender bias in employment
   - Understanding results

3. **[Data Tiers Explained](beginner/03_understanding_data_tiers.ipynb)** (45 mins)

   - Lite vs comprehensive analysis
   - When to use each tier
   - Performance considerations

4. **[Interpreting Results](beginner/04_interpreting_results.ipynb)** (45 mins)
   - Statistical significance
   - Effect sizes
   - Practical implications

### 🏗️ **Intermediate Path** (4-5 hours)

**Perfect for**: Developers, data scientists, platform users

1. **[Custom Bias Categories](intermediate/05_custom_bias_categories.ipynb)** (60 mins)

   - Creating custom bias definitions
   - Domain-specific bias detection
   - Validation strategies

2. **[Feature Extraction Deep Dive](intermediate/06_feature_extraction.ipynb)** (60 mins)

   - Sentiment analysis
   - Toxicity detection
   - Custom feature extractors

3. **[Statistical Analysis](intermediate/07_statistical_analysis.ipynb)** (60 mins)

   - Hypothesis testing
   - Multiple comparisons
   - Confidence intervals

4. **[Web Platform Usage](intermediate/08_web_platform_usage.ipynb)** (60 mins)
   - API integration
   - Benchmark management
   - Experiment workflows

### 🚀 **Advanced Path** (6+ hours)

**Perfect for**: Researchers, ML engineers, platform developers

1. **[Custom Models Integration](advanced/09_custom_models.ipynb)** (90 mins)

   - Local LLM integration
   - Custom API endpoints
   - Performance optimization

2. **[Pipeline Optimization](advanced/10_pipeline_optimization.ipynb)** (90 mins)

   - Parallel processing
   - Memory management
   - Caching strategies

3. **[Batch Processing](advanced/11_batch_processing.ipynb)** (90 mins)

   - Large-scale analysis
   - Distributed computing
   - Result aggregation

4. **[Research Applications](advanced/12_research_applications.ipynb)** (120 mins)
   - Academic research workflows
   - Publication-ready results
   - Reproducibility guidelines

## 📖 Tutorial Descriptions

### Beginner Tutorials

#### 1. Introduction to SAGED (`01_introduction_to_saged.ipynb`)

**What you'll learn:**

- Core concepts of AI bias
- SAGED methodology (Scrape → Assemble → Generate → Extract → Diagnose)
- Installation and environment setup
- First bias detection example

**Prerequisites:** Basic Python knowledge

```python
# Example from the tutorial
from saged import SAGEDData

# Initialize your first bias detector
saged = SAGEDData(
    domain="employment",
    concept="gender",
    data_tier="lite"
)

# Run bias detection
results = saged.run_full_pipeline()
print(f"Bias score: {results['bias_metrics']['overall_bias_score']}")
```

#### 2. Basic Bias Detection (`02_basic_bias_detection.ipynb`)

**What you'll learn:**

- Step-by-step bias analysis
- Gender bias in employment scenarios
- Interpreting bias scores
- Visualizing results

**Hands-on exercises:**

- Detect bias in job descriptions
- Compare male vs female sentiment scores
- Generate bias reports

#### 3. Understanding Data Tiers (`03_understanding_data_tiers.ipynb`)

**What you'll learn:**

- 6 data tiers: lite → questions
- Trade-offs: speed vs comprehensiveness
- Choosing the right tier for your use case
- Performance benchmarks

**Comparison table:**
| Tier | Samples | Time | Use Case |
|------|---------|------|----------|
| Lite | 50 | 2 min | Quick exploration |
| Keywords | 200 | 8 min | Standard analysis |
| Questions | 2000+ | 60+ min | Research-grade |

### Intermediate Tutorials

#### 5. Custom Bias Categories (`05_custom_bias_categories.ipynb`)

**What you'll learn:**

- Define your own bias categories
- Create domain-specific tests
- Validate custom bias definitions
- Industry-specific examples

```python
# Create custom bias category
custom_bias = {
    "groups": ["experienced", "junior", "senior"],
    "keywords": {
        "experienced": ["veteran", "seasoned", "expert"],
        "junior": ["entry-level", "new", "fresh"],
        "senior": ["senior", "lead", "principal"]
    }
}

saged = SAGEDData(concept="custom", custom_config=custom_bias)
```

#### 6. Feature Extraction (`06_feature_extraction.ipynb`)

**What you'll learn:**

- Built-in extractors (sentiment, toxicity, embeddings)
- Custom feature development
- Feature engineering for bias detection
- Combining multiple features

#### 7. Statistical Analysis (`07_statistical_analysis.ipynb`)

**What you'll learn:**

- P-values and significance testing
- Effect size calculation (Cohen's d)
- Multiple hypothesis correction
- Confidence intervals
- Power analysis

### Advanced Tutorials

#### 9. Custom Models (`09_custom_models.ipynb`)

**What you'll learn:**

- Integrate local LLMs (Llama, Mistral)
- Custom API endpoints
- Model comparison studies
- Performance optimization

```python
from saged.models import CustomModel

class LocalLlama(CustomModel):
    def __init__(self, model_path):
        self.model = load_llama_model(model_path)

    def generate(self, prompt):
        return self.model.generate(prompt)

# Use custom model
saged = SAGEDData(concept="gender")
saged.set_model(LocalLlama("path/to/llama"))
```

#### 10. Pipeline Optimization (`10_pipeline_optimization.ipynb`)

**What you'll learn:**

- Parallel processing techniques
- Memory optimization strategies
- Caching for faster re-runs
- Profiling and performance monitoring

## 🌍 Real-World Examples

### Employment Bias Study (`examples/employment_bias_study.ipynb`)

**Scenario:** Analyzing bias in job descriptions and hiring processes

**What's covered:**

- Fortune 500 job posting analysis
- Gender bias in technical roles
- Salary prediction bias
- Actionable recommendations

**Results preview:**

- 23% bias detected in technical job descriptions
- Female-coded language reduces application rates by 15%
- Specific keywords that introduce bias

### Healthcare Bias Analysis (`examples/healthcare_bias_analysis.ipynb`)

**Scenario:** Detecting bias in medical AI systems

**What's covered:**

- Symptom description bias
- Treatment recommendation analysis
- Demographic disparities
- Ethical implications

### Education Bias Detection (`examples/education_bias_detection.ipynb`)

**Scenario:** Bias in educational content and assessments

**What's covered:**

- Textbook content analysis
- Grading bias detection
- Student evaluation fairness
- Curriculum recommendations

## 🔧 Interactive Features

### Code Snippets

All tutorials include copy-paste ready code:

```python
# Quick bias check
from saged import quick_bias_check

bias_score = quick_bias_check(
    text="Describe an ideal software engineer",
    category="gender"
)

print(f"Bias detected: {bias_score:.2%}")
```

### Visualization Examples

```python
import matplotlib.pyplot as plt
from saged.visualization import plot_bias_comparison

# Compare bias across categories
plot_bias_comparison(results, categories=["gender", "race", "age"])
plt.show()
```

### Interactive Widgets

```python
from ipywidgets import interact, IntSlider

@interact(data_tier=["lite", "keywords", "questions"])
def run_bias_analysis(data_tier):
    saged = SAGEDData(data_tier=data_tier)
    results = saged.run_full_pipeline()
    return results["bias_metrics"]["overall_bias_score"]
```

## 📊 Datasets

### Included Sample Datasets

#### Employment Prompts (`datasets/employment_prompts.csv`)

- 500 job-related prompts
- Multiple bias categories
- Real-world scenarios

#### Healthcare Scenarios (`datasets/healthcare_scenarios.json`)

- Medical case studies
- Patient descriptions
- Treatment scenarios

#### Social Media Posts (`datasets/social_media_posts.csv`)

- Anonymized social media content
- Demographic annotations
- Bias labels

### Using Custom Datasets

```python
import pandas as pd

# Load your own data
custom_data = pd.read_csv("your_dataset.csv")

# Convert to SAGED format
saged_prompts = saged.prepare_custom_data(
    custom_data,
    text_column="content",
    group_column="demographic"
)

# Run analysis
results = saged.run_analysis(saged_prompts)
```

## 🎯 Learning Objectives

By completing these tutorials, you will:

### **Knowledge Goals**

- ✅ Understand AI bias types and implications
- ✅ Master the SAGED methodology
- ✅ Know when to use different data tiers
- ✅ Interpret statistical significance correctly

### **Technical Skills**

- ✅ Configure and run bias detection pipelines
- ✅ Create custom bias categories
- ✅ Integrate with web platform APIs
- ✅ Optimize performance for large datasets

### **Practical Applications**

- ✅ Conduct real-world bias audits
- ✅ Generate actionable bias reports
- ✅ Design bias mitigation strategies
- ✅ Publish reproducible research

## 🛠️ Prerequisites

### Required Knowledge

- **Python**: Basic to intermediate (variables, functions, classes)
- **Statistics**: Descriptive statistics, hypothesis testing
- **Data Science**: Pandas, NumPy basics

### Optional but Helpful

- **Machine Learning**: Understanding of LLMs and NLP
- **Web Development**: For API integration tutorials
- **Research Methods**: For academic application examples

### Software Requirements

```bash
# Core requirements
python>=3.10
jupyter>=1.0
pandas>=1.3
numpy>=1.20
matplotlib>=3.5

# SAGED-specific
saged>=1.0  # This package
requests>=2.25  # For API calls
scikit-learn>=1.0  # For ML features

# Optional enhancements
plotly>=5.0  # Interactive plots
ipywidgets>=7.6  # Interactive widgets
seaborn>=0.11  # Advanced plotting
```

## 🔄 Tutorial Updates

Tutorials are regularly updated to reflect:

- **Latest SAGED features**
- **New bias detection methods**
- **Updated datasets**
- **Community feedback**
- **Research developments**

### Version History

- **v1.0** (Jan 2024): Initial tutorial suite
- **v1.1** (Feb 2024): Added custom model integration
- **v1.2** (Mar 2024): Enhanced visualization examples
- **v1.3** (Current): Added real-world case studies

## 🤝 Contributing

### Adding New Tutorials

1. **Fork** the repository
2. **Create** new notebook in appropriate folder
3. **Follow** naming convention: `##_descriptive_name.ipynb`
4. **Include** learning objectives and prerequisites
5. **Test** all code cells
6. **Submit** pull request

### Tutorial Guidelines

- **Clear learning objectives** at the start
- **Step-by-step explanations** with code
- **Real-world examples** and use cases
- **Exercises** for hands-on learning
- **Further reading** suggestions
- **Estimated completion time**

### Feedback and Improvements

- **Issues**: Report problems or suggest improvements
- **Discussions**: Share use cases and experiences
- **Pull Requests**: Contribute fixes and enhancements

## 📞 Support

### Getting Help

- **Documentation**: [Main README](../README.md)
- **API Reference**: [Backend Docs](../app/backend/README.md)
- **Community**: GitHub Discussions
- **Issues**: GitHub Issues for bugs

### Common Issues

| Problem             | Solution                               |
| ------------------- | -------------------------------------- |
| Jupyter won't start | `pip install jupyter notebook`         |
| SAGED import fails  | Check installation: `pip install -e .` |
| Memory errors       | Use smaller data tier or add RAM       |
| Slow execution      | Enable caching and parallel processing |

---

**Ready to start learning?** Open [01_introduction_to_saged.ipynb](beginner/01_introduction_to_saged.ipynb) and begin your journey into AI bias detection!

For more information, see the [main README](../README.md) or the [SAGED core documentation](../saged/README.md).
