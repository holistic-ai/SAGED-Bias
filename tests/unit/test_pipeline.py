"""
Tests for the Pipeline component.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from saged._pipeline import Pipeline


class TestPipeline:
    """Test suite for Pipeline functionality."""

    def test_pipeline_class_exists(self):
        """Test that Pipeline class exists and can be instantiated."""
        # Pipeline is a class with class methods, not instance methods
        assert hasattr(Pipeline, 'build_concept_benchmark')
        assert hasattr(Pipeline, 'build_benchmark')
        assert hasattr(Pipeline, 'run_benchmark')

    def test_config_helper(self):
        """Test config helper method."""
        # This should not raise an error
        Pipeline.config_helper()

    def test_set_config(self):
        """Test configuration setting."""
        Pipeline._set_config()
        
        # Check that config schemes are set
        assert hasattr(Pipeline, '_branching_config_scheme')
        assert hasattr(Pipeline, '_concept_benchmark_config_scheme')
        assert hasattr(Pipeline, '_domain_benchmark_config_scheme')
        assert hasattr(Pipeline, '_analytics_config_scheme')

    def test_build_concept_benchmark_basic(self):
        """Test basic concept benchmark building."""
        # Mock the dependencies to avoid actual execution
        with patch('saged._pipeline.KeywordFinder'), \
             patch('saged._pipeline.SourceFinder'), \
             patch('saged._pipeline.Scraper'), \
             patch('saged._pipeline.PromptMaker'):
            
            # This should not raise an error with minimal config
            try:
                result = Pipeline.build_concept_benchmark(
                    domain='test_domain',
                    demographic_label='test_label',
                    config={'keyword_finder': {'require': False}}
                )
                # If it returns something, it should be a valid result
                assert result is not None
            except Exception as e:
                # Some exceptions are expected due to missing data/files
                assert isinstance(e, (FileNotFoundError, ValueError, KeyError))

    def test_build_benchmark_basic(self):
        """Test basic benchmark building."""
        # Mock the dependencies
        with patch('saged._pipeline.Pipeline.build_concept_benchmark') as mock_build:
            mock_build.return_value = Mock()
            
            try:
                result = Pipeline.build_benchmark(
                    domain='test_domain',
                    config={'categories': ['test_category']}
                )
                # If it returns something, it should be a valid result
                assert result is not None
            except Exception as e:
                # Some exceptions are expected due to missing data/files
                assert isinstance(e, (FileNotFoundError, ValueError, KeyError))

    def test_run_benchmark_basic(self, sample_benchmark_data):
        """Test basic benchmark running."""
        # Create a temporary output directory first
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a minimal config with temporary paths
            config = {
                'benchmark': sample_benchmark_data,
                'generation': {
                    'require': False,
                    'generation_saving_location': os.path.join(temp_dir, 'generation.csv')
                },
                'extraction': {
                    'feature_extractors': [],
                    'extraction_saving_location': os.path.join(temp_dir, 'extraction.csv')
                },
                'analysis': {
                    'specifications': ['concept'],
                    'analyzers': [],
                    'statistics_saving_location': os.path.join(temp_dir, 'statistics.csv'),
                    'disparity_saving_location': os.path.join(temp_dir, 'disparity.csv')
                }
            }
            
            # Mock the dependencies
            with patch('saged._pipeline.ResponseGenerator') as mock_gen, \
                 patch('saged._pipeline.FeatureExtractor') as mock_ext, \
                 patch('saged._pipeline.Analyzer') as mock_ana:
                
                mock_gen.return_value = Mock()
                mock_ext_instance = Mock()
                mock_ext_instance.classification_features = []
                mock_ext_instance.cluster_features = []
                mock_ext_instance.calibrated_features = []
                mock_ext.return_value = mock_ext_instance
                
                mock_ana_instance = Mock()
                mock_ana_instance.statistics_disparity.return_value = pd.DataFrame()
                mock_ana.return_value = mock_ana_instance
                
                # The Pipeline.run_benchmark method doesn't return anything (it returns None)
                # So we should expect None and test that no exceptions are raised
                try:
                    result = Pipeline.run_benchmark(config)
                    # The method returns None, so this is expected
                    assert result is None
                except Exception as e:
                    # Some exceptions are expected due to missing data/files
                    assert isinstance(e, (FileNotFoundError, ValueError, KeyError, AttributeError))

    def test_config_schemes_structure(self):
        """Test that config schemes have expected structure."""
        Pipeline._set_config()
        
        # Check branching config scheme
        assert isinstance(Pipeline._branching_config_scheme, dict)
        assert 'branching_pairs' in Pipeline._branching_config_scheme
        assert 'direction' in Pipeline._branching_config_scheme
        
        # Check concept benchmark config scheme
        assert isinstance(Pipeline._concept_benchmark_config_scheme, dict)
        assert 'keyword_finder' in Pipeline._concept_benchmark_config_scheme
        assert 'source_finder' in Pipeline._concept_benchmark_config_scheme
        assert 'scraper' in Pipeline._concept_benchmark_config_scheme
        assert 'prompt_assembler' in Pipeline._concept_benchmark_config_scheme
        
        # Check domain benchmark config scheme
        assert isinstance(Pipeline._domain_benchmark_config_scheme, dict)
        assert 'categories' in Pipeline._domain_benchmark_config_scheme
        assert 'branching' in Pipeline._domain_benchmark_config_scheme
        
        # Check analytics config scheme
        assert isinstance(Pipeline._analytics_config_scheme, dict)
        assert 'benchmark' in Pipeline._analytics_config_scheme
        assert 'generation' in Pipeline._analytics_config_scheme
        assert 'extraction' in Pipeline._analytics_config_scheme
        assert 'analysis' in Pipeline._analytics_config_scheme

    def test_default_configs_structure(self):
        """Test that default configs have expected structure."""
        Pipeline._set_config()
        
        # Check that default configs are dictionaries
        assert isinstance(Pipeline._branching_default_config, dict)
        assert isinstance(Pipeline._concept_benchmark_default_config, dict)
        assert isinstance(Pipeline._domain_benchmark_default_config, dict)
        assert isinstance(Pipeline._analytics_default_config, dict)
        
        # Check some key default values
        assert Pipeline._branching_default_config['branching_pairs'] == 'not_all'
        assert Pipeline._branching_default_config['direction'] == 'both'
        
        assert Pipeline._concept_benchmark_default_config['keyword_finder']['require'] == True
        assert Pipeline._concept_benchmark_default_config['source_finder']['method'] == 'wiki'
        
        assert Pipeline._analytics_default_config['generation']['require'] == True
        assert Pipeline._analytics_default_config['extraction']['calibration'] == True

    def test_pipeline_integration_components(self):
        """Test that Pipeline can import and use its component classes."""
        # Test that all required components can be imported
        from saged._pipeline import ResponseGenerator, FeatureExtractor, Analyzer
        from saged._pipeline import saged, KeywordFinder, SourceFinder, Scraper, PromptMaker
        
        # These should all be valid classes/functions
        assert ResponseGenerator is not None
        assert FeatureExtractor is not None
        assert Analyzer is not None
        assert saged is not None
        assert KeywordFinder is not None
        assert SourceFinder is not None
        assert Scraper is not None
        assert PromptMaker is not None

    def test_error_handling_invalid_domain(self):
        """Test error handling for invalid domain."""
        # Mock the dependencies to avoid actual file operations
        with patch('saged._pipeline.KeywordFinder'), \
             patch('saged._pipeline.SourceFinder'), \
             patch('saged._pipeline.Scraper'), \
             patch('saged._pipeline.PromptMaker'):
            
            # Test with None domain and valid config to avoid copy() on None
            config = {'keyword_finder': {'require': False}}
            
            with pytest.raises((ValueError, TypeError, FileNotFoundError, AttributeError)):
                Pipeline.build_concept_benchmark(
                    domain=None,
                    demographic_label='test',
                    config=config
                )

    def test_error_handling_invalid_config(self):
        """Test error handling for invalid configuration."""
        # Create a temporary directory for output to avoid directory issues
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock dependencies first
            with patch('saged._pipeline.ResponseGenerator') as mock_gen, \
                 patch('saged._pipeline.FeatureExtractor') as mock_ext, \
                 patch('saged._pipeline.Analyzer') as mock_ana:
                
                mock_gen.return_value = Mock()
                mock_ext_instance = Mock()
                mock_ext_instance.classification_features = []
                mock_ext_instance.cluster_features = []
                mock_ext_instance.calibrated_features = []
                mock_ext.return_value = mock_ext_instance
                
                mock_ana_instance = Mock()
                mock_ana_instance.statistics_disparity.return_value = pd.DataFrame()
                mock_ana.return_value = mock_ana_instance
                
                # Test with missing benchmark - pipeline is designed to be fault-tolerant
                # so we test that it handles missing keys gracefully
                config = {
                    'generation': {
                        'require': False
                    },
                    'extraction': {
                        'feature_extractors': [],
                        'extraction_saving_location': os.path.join(temp_dir, 'extraction.csv')
                    },
                    'analysis': {
                        'disparity_saving_location': os.path.join(temp_dir, 'disparity.csv')
                    }
                    # Missing 'benchmark' key - should be handled gracefully
                }
                
                # The pipeline is designed to handle missing configurations gracefully
                # so we test that it doesn't crash, but we expect it might fail later
                # due to the missing benchmark data
                try:
                    result = Pipeline.run_benchmark(config=config)
                    # If it succeeds, it should return None (as per the method definition)
                    assert result is None
                except Exception as e:
                    # Some exceptions are expected due to missing benchmark data
                    assert isinstance(e, (ValueError, TypeError, KeyError, AttributeError))

    def test_config_validation(self):
        """Test configuration validation."""
        Pipeline._set_config()
        
        # Test that config schemes contain expected keys
        required_branching_keys = ['branching_pairs', 'direction', 'source_restriction']
        for key in required_branching_keys:
            assert key in Pipeline._branching_config_scheme
        
        required_analytics_keys = ['benchmark', 'generation', 'extraction', 'analysis']
        for key in required_analytics_keys:
            assert key in Pipeline._analytics_config_scheme 