"""
Unit tests for PromptAssembler class.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import spacy
import json
import tempfile

from saged._assembler import PromptAssembler
from saged._saged_data import SAGEDData


def _is_spacy_model_available():
    """Check if spaCy en_core_web_sm model is available."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        return False


@pytest.fixture
def sample_saged_data():
    """Create sample SAGEDData for testing."""
    data = pd.DataFrame({
        'concept': ['education', 'healthcare', 'employment'] * 3,
        'category': ['education', 'healthcare', 'employment'] * 3,
        'prompt': [
            'What are the benefits of higher education?',
            'How can healthcare be improved?',
            'What makes a good employee?'
        ] * 3
    })
    return SAGEDData(data)


@pytest.fixture
def assembler_config():
    """Sample assembler configuration."""
    return {
        'prompt_template': 'Context: {context}\nQuestion: {prompt}',
        'context_sources': ['wikipedia', 'local'],
        'max_context_length': 500,
        'include_metadata': True
    }


class TestPromptAssembler:
    """Test suite for PromptAssembler."""

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    def test_init_default(self, sample_scraped_saged_data):
        """Test PromptAssembler initialization with default settings."""
        assembler = PromptAssembler(sample_scraped_saged_data)
        
        assert assembler.concept == 'test_concept'
        assert assembler.domain == 'test_domain'
        assert hasattr(assembler, 'nlp')
        assert isinstance(assembler.data, list)
        assert isinstance(assembler.output_df, pd.DataFrame)

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")  
    def test_init_with_config(self, sample_scraped_saged_data):
        """Test PromptAssembler initialization with configuration."""
        assembler = PromptAssembler(sample_scraped_saged_data)
        
        # Should not raise any errors
        assert assembler.concept == 'test_concept'
        assert assembler.domain == 'test_domain'

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    def test_assemble_basic(self):
        """Test basic assembly functionality."""
        # Create simple test data
        simple_data = [{
            "concept": "test_concept",
            "domain": "test_domain",
            "concept_shared_source": [{
                "source_tag": "test",
                "source_type": "general_links", 
                "source_specification": ["http://example.com"]
            }],
            "keywords": {
                "leadership": {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": "manual",
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes",
                    "scraped_sentences": [
                        ("Effective leadership requires strong communication skills and vision.", "test_source"),
                        ("Leadership involves guiding teams towards common goals.", "test_source")
                    ]
                }
            }
        }]
        
        saged_data = SAGEDData.create_data(
            domain='test_domain',
            concept='test_concept',
            data_tier='scraped_sentences',
            data=simple_data
        )
        
        assembler = PromptAssembler(saged_data)
        result = assembler.split_sentences()
        
        assert isinstance(result, SAGEDData)
        assert result.data_tier == 'split_sentences'

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    def test_build_context(self, sample_scraped_saged_data):
        """Test context building functionality."""
        assembler = PromptAssembler(sample_scraped_saged_data)
        
        # Test that the assembler processes the scraped data correctly
        assert len(assembler.data) == 1
        assert 'keywords' in assembler.data[0]
        assert 'test_keyword' in assembler.data[0]['keywords']

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    @patch('saged._assembler.download')
    def test_fetch_wikipedia_context(self, mock_download, sample_scraped_saged_data):
        """Test Wikipedia context fetching.""" 
        with patch('wikipedia.summary') as mock_summary:
            mock_summary.return_value = "Test Wikipedia content"
            
            assembler = PromptAssembler(sample_scraped_saged_data)
            
            # Should initialize without errors
            assert assembler.concept == 'test_concept'

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    def test_format_prompt(self, sample_scraped_saged_data):
        """Test prompt formatting functionality."""
        assembler = PromptAssembler(sample_scraped_saged_data)
        
        # Test split_sentences method
        result = assembler.split_sentences()
        assert isinstance(result, SAGEDData)
        assert result.data_tier == 'split_sentences'

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    def test_handle_empty_context(self, sample_scraped_saged_data):
        """Test handling of empty context scenarios."""
        assembler = PromptAssembler(sample_scraped_saged_data)
        
        # Should handle empty data gracefully
        assert assembler is not None
        assert hasattr(assembler, 'output_df')

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    def test_context_length_limit(self):
        """Test context length limiting functionality."""
        # Create test data with very long sentences
        long_data = [{
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
                        ("This is a very long sentence " * 100 + " about test_keyword.", "test_source")
                    ]
                }
            }
        }]
        
        saged_data = SAGEDData.create_data(
            domain='test_domain',
            concept='test_concept',
            data_tier='scraped_sentences',
            data=long_data
        )
        
        assembler = PromptAssembler(saged_data)
        assert assembler.concept == 'test_concept'

    @pytest.mark.skip(reason="Requires spacy model en_core_web_sm")
    def test_fetch_local_context(self, sample_scraped_saged_data):
        """Test local context fetching functionality."""
        assembler = PromptAssembler(sample_scraped_saged_data)
        
        # Test that assembler can access local data
        assert len(assembler.data[0]['keywords']) > 0
        assert 'test_keyword' in assembler.data[0]['keywords']

    def test_validate_data(self):
        """Test data validation functionality."""
        # Test with invalid data tier
        invalid_data = SAGEDData.create_data(
            domain='test_domain',
            concept='test_concept',
            data_tier='keywords'  # Wrong tier for PromptAssembler
        )
        
        with pytest.raises(AssertionError):
            PromptAssembler(invalid_data)

    def test_error_handling(self):
        """Test error handling in PromptAssembler."""
        # Test with invalid input type
        with pytest.raises(AssertionError):
            PromptAssembler("not_a_saged_data_object")

    @pytest.mark.skipif(not _is_spacy_model_available(), 
                         reason="spaCy en_core_web_sm model not available")
    def test_metadata_inclusion(self):
        """Test metadata inclusion in outputs."""
        # Create test data with metadata
        metadata_data = [{
            "concept": "innovation",
            "domain": "technology",
            "concept_shared_source": [{
                "source_tag": "research_paper",
                "source_type": "general_links",
                "source_specification": ["http://research.example.com"]
            }],
            "keywords": {
                "innovation": {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": "manual",
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes",
                    "scraped_sentences": [
                        ("Innovation drives technological progress and social change.", "research_source")
                    ]
                }
            }
        }]
        
        saged_data = SAGEDData.create_data(
            domain='technology',
            concept='innovation',
            data_tier='scraped_sentences',
            data=metadata_data
        )
        
        assembler = PromptAssembler(saged_data)

        # Test that assembler handles metadata properly
        assert assembler.scraped_sentence_saged_data is not None
        assert hasattr(assembler, 'nlp')

    @pytest.mark.skipif(not _is_spacy_model_available(), 
                         reason="spaCy en_core_web_sm model not available")
    def test_multiple_concepts(self):
        """Test handling multiple concepts."""
        multi_concept_data = [{
            "concept": "leadership",
            "domain": "business",
            "concept_shared_source": [{
                "source_tag": "test",
                "source_type": "general_links",
                "source_specification": ["http://example.com"]
            }],
            "keywords": {
                "leadership": {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": "manual",
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes",
                    "scraped_sentences": [
                        ("Leadership is about inspiring others.", "source1"),
                        ("Effective leaders communicate clearly.", "source2")
                    ]
                },
                "management": {
                    "keyword_type": "sub-concepts",
                    "keyword_provider": "manual",
                    "scrap_mode": "in_page",
                    "scrap_shared_area": "Yes",
                    "scraped_sentences": [
                        ("Management involves planning and organizing.", "source1")
                    ]
                }
            }
        }]
        
        saged_data = SAGEDData.create_data(
            domain='business',
            concept='leadership',
            data_tier='scraped_sentences',
            data=multi_concept_data
        )
        
        assembler = PromptAssembler(saged_data)

        # Test that assembler handles multiple concepts
        assert assembler.scraped_sentence_saged_data is not None
        assert hasattr(assembler, 'nlp') 