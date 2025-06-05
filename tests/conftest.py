"""
Shared fixtures for SAGED tests.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
from pathlib import Path
import tempfile

from saged._saged_data import SAGEDData


@pytest.fixture
def sample_benchmark_data():
    """Sample DataFrame with benchmark data for testing."""
    return pd.DataFrame({
        'concept': ['leadership', 'leadership', 'innovation', 'innovation', 'teamwork', 'teamwork'],
        'domain': ['business', 'politics', 'technology', 'science', 'sports', 'education'],
        'baseline': [
            'A leader guides and motivates people to achieve goals.',
            'Political leaders make decisions affecting entire communities.',
            'Innovation involves creating new solutions to existing problems.',
            'Scientific innovation advances human knowledge and capabilities.',
            'Teamwork combines individual strengths for collective success.',
            'Educational teams collaborate to enhance student learning.'
        ],
        'LLM': [
            'Effective leaders inspire teams to achieve common goals.',
            'Political leadership requires balancing diverse interests.',
            'Innovation emerges from creative problem-solving approaches.',
            'Scientific breakthroughs result from rigorous experimentation.',
            'Successful teamwork depends on trust and communication.',
            'Educational partnerships foster collaborative learning.'
        ],
        'baseline_sentiment_score': [0.7, 0.8, 0.6, 0.75, 0.85, 0.8],
        'LLM_sentiment_score': [0.65, 0.85, 0.7, 0.8, 0.9, 0.75],
        'baseline_toxicity_score': [0.1, 0.05, 0.15, 0.08, 0.02, 0.06],
        'LLM_toxicity_score': [0.12, 0.03, 0.1, 0.06, 0.01, 0.08],
        'source_tag': ['wikipedia', 'news', 'blog', 'journal', 'article', 'book']
    })


@pytest.fixture
def sample_benchmark_data_for_diagnoser():
    """Sample DataFrame with benchmark data specifically for DisparityDiagnoser testing."""
    return pd.DataFrame({
        'keyword': ['leadership', 'leadership', 'innovation', 'innovation', 'teamwork', 'teamwork'],
        'concept': ['leadership', 'leadership', 'innovation', 'innovation', 'teamwork', 'teamwork'],
        'domain': ['business', 'politics', 'technology', 'science', 'sports', 'education'],
        'baseline': [
            'A leader guides and motivates people to achieve goals.',
            'Political leaders make decisions affecting entire communities.',
            'Innovation involves creating new solutions to existing problems.',
            'Scientific innovation advances human knowledge and capabilities.',
            'Teamwork combines individual strengths for collective success.',
            'Educational teams collaborate to enhance student learning.'
        ],
        'LLM': [
            'Effective leaders inspire teams to achieve common goals.',
            'Political leadership requires balancing diverse interests.',
            'Innovation emerges from creative problem-solving approaches.',
            'Scientific breakthroughs result from rigorous experimentation.',
            'Successful teamwork depends on trust and communication.',
            'Educational partnerships foster collaborative learning.'
        ],
        'baseline_sentiment_score': [0.7, 0.8, 0.6, 0.75, 0.85, 0.8],
        'LLM_sentiment_score': [0.65, 0.85, 0.7, 0.8, 0.9, 0.75],
        'baseline_toxicity_score': [0.1, 0.05, 0.15, 0.08, 0.02, 0.06],
        'LLM_toxicity_score': [0.12, 0.03, 0.1, 0.06, 0.01, 0.08],
        'source_tag': ['wikipedia', 'news', 'blog', 'journal', 'article', 'book']
    })


@pytest.fixture 
def sample_scraped_saged_data():
    """Sample SAGEDData instance with scraped sentences data tier."""
    return SAGEDData.create_data(
        domain='test_domain',
        concept='test_concept', 
        data_tier='scraped_sentences',
        data=[{
            "concept": "test_concept",
            "domain": "test_domain",
            "concept_shared_source": [{
                "source_tag": "test",
                "source_type": "general_links", 
                "source_specification": ["http://example.com"]
            }],
            "keywords": {
                "test_keyword": {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": "manual",
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes",
                    "scraped_sentences": [
                        ("This is a test sentence about test_keyword.", "test_source"),
                        ("Another sentence with test_keyword content.", "test_source")
                    ]
                }
            }
        }]
    )


@pytest.fixture
def sample_generations_data():
    """Sample DataFrame with multiple generations for testing."""
    return pd.DataFrame({
        'concept': ['leadership'] * 4,
        'domain': ['business'] * 4,
        'baseline': ['A leader guides people effectively.'] * 4,
        'LLM': [
            'Effective leaders inspire and motivate teams.',
            'Leadership requires vision and communication.',
            'Great leaders build trust and collaboration.',
            'Successful leaders adapt to changing situations.'
        ],
        'optimist': [
            'Outstanding leaders create amazing opportunities!',
            'Visionary leaders transform organizations brilliantly!',
            'Exceptional leaders build incredible teams!',
            'Dynamic leaders achieve remarkable success!'
        ],
        'realist': [
            'Leaders face challenges and work through them.',
            'Leadership involves practical decision-making.',
            'Effective leaders balance multiple priorities.',
            'Leaders must adapt to real-world constraints.'
        ]
    })


@pytest.fixture
def sample_extractions_data():
    """Sample DataFrame with extracted features."""
    return pd.DataFrame({
        'concept': ['leadership', 'innovation', 'teamwork'],
        'domain': ['business', 'technology', 'sports'],
        'text': [
            'Leadership requires vision and communication',
            'Innovation drives technological progress',
            'Teamwork enhances performance outcomes'
        ],
        'sentiment_score': [0.8, 0.7, 0.85],
        'toxicity_score': [0.05, 0.1, 0.02],
        'stereotype_score': [0.15, 0.08, 0.12],
        'group': ['A', 'B', 'A']
    })


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        'choices': [{
            'message': {
                'content': 'This is a mock response from the language model.'
            }
        }],
        'usage': {
            'prompt_tokens': 50,
            'completion_tokens': 25,
            'total_tokens': 75
        }
    }


@pytest.fixture
def sample_system_prompts():
    """Sample system prompts for testing."""
    return {
        'optimist': 'You are an optimistic assistant who always looks on the bright side.',
        'realist': 'You are a realistic assistant who provides balanced perspectives.',
        'empathetic': 'You are an empathetic assistant who considers emotional impacts.',
        'cautious': 'You are a cautious assistant who considers risks and challenges.',
        'critical': 'You are a critical assistant who analyzes things thoroughly.'
    }


@pytest.fixture
def temp_directory():
    """Temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_network_access():
    """Mock network access to prevent actual API calls during tests."""
    with pytest.MonkeyPatch.context() as m:
        # Mock requests
        m.setattr("requests.get", Mock())
        m.setattr("requests.post", Mock())
        
        # Mock OpenAI client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Mock response"))]
        )
        m.setattr("openai.OpenAI", Mock(return_value=mock_client))
        
        # Mock Wikipedia API
        m.setattr("wikipedia.summary", Mock(return_value="Mock Wikipedia summary"))
        m.setattr("wikipedia.search", Mock(return_value=["Mock Result"]))
        
        yield


@pytest.fixture
def sample_pipeline_config():
    """Sample pipeline configuration for testing."""
    return {
        'batch_size': 32,
        'max_retries': 3,
        'timeout': 300,
        'enable_caching': True,
        'enable_progress_tracking': True
    }


@pytest.fixture
def sample_bias_report_data():
    """Sample data for bias report generation testing."""
    return pd.DataFrame({
        'sentiment_score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'toxicity_score': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'domain': ['tech', 'tech', 'health', 'health', 'finance', 'finance', 'education', 'education'],
        'group': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })


@pytest.fixture
def mock_model_pipeline():
    """Mock transformers pipeline for testing."""
    mock_pipeline = Mock()
    mock_pipeline.return_value = [
        {'label': 'positive', 'score': 0.7},
        {'label': 'negative', 'score': 0.3}
    ]
    return mock_pipeline


@pytest.fixture
def sample_weights_config():
    """Sample weights configuration for MPF testing."""
    return {
        "timestamp": "20250115_143649",
        "mitigation_type": "wasserstein_weighted", 
        "feature": "sentiment",
        "baseline_generation": "baseline",
        "component_generations": ["optimist", "realist"],
        "perspectives": {
            "optimist": {
                "system_prompt": "You are an optimistic assistant.",
                "weights": {
                    "leadership": 0.7,
                    "innovation": 0.3,
                    "teamwork": 0.5
                }
            },
            "realist": {
                "system_prompt": "You are a realistic assistant.",
                "weights": {
                    "leadership": 0.3,
                    "innovation": 0.7,
                    "teamwork": 0.5
                }
            }
        }
    }


@pytest.fixture 
def sample_benchmark_data_with_prompts():
    """Sample DataFrame with benchmark data including prompts column for ResponseGenerator testing."""
    return pd.DataFrame({
        'concept': ['leadership', 'leadership', 'innovation', 'innovation', 'teamwork', 'teamwork'],
        'domain': ['business', 'politics', 'technology', 'science', 'sports', 'education'],
        'prompts': [
            'Write about leadership in business contexts',
            'Discuss political leadership challenges',
            'Describe innovation in technology',
            'Explain scientific innovation processes',
            'Analyze teamwork in sports',
            'Examine educational teamwork strategies'
        ],
        'baseline': [
            'A leader guides and motivates people to achieve goals.',
            'Political leaders make decisions affecting entire communities.',
            'Innovation involves creating new solutions to existing problems.',
            'Scientific innovation advances human knowledge and capabilities.',
            'Teamwork combines individual strengths for collective success.',
            'Educational teams collaborate to enhance student learning.'
        ],
        'LLM': [
            'Effective leaders inspire teams to achieve common goals.',
            'Political leadership requires balancing diverse interests.',
            'Innovation emerges from creative problem-solving approaches.',
            'Scientific breakthroughs result from rigorous experimentation.',
            'Successful teamwork depends on trust and communication.',
            'Educational partnerships foster collaborative learning.'
        ],
        'baseline_sentiment_score': [0.7, 0.8, 0.6, 0.75, 0.85, 0.8],
        'LLM_sentiment_score': [0.65, 0.85, 0.7, 0.8, 0.9, 0.75],
        'baseline_toxicity_score': [0.1, 0.05, 0.15, 0.08, 0.02, 0.06],
        'LLM_toxicity_score': [0.12, 0.03, 0.1, 0.06, 0.01, 0.08],
        'source_tag': ['wikipedia', 'news', 'blog', 'journal', 'article', 'book']
    }) 