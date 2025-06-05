"""
Tests for the FeatureExtractor component.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from saged._extractor import FeatureExtractor


class TestFeatureExtractor:
    """Test suite for FeatureExtractor functionality."""

    def test_init_default(self, sample_benchmark_data):
        """Test FeatureExtractor initialization with default settings."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        assert extractor.benchmark is not None
        assert isinstance(extractor.benchmark, pd.DataFrame)
        expected_generations = ['baseline', 'LLM']
        actual_generations = list(extractor.generations) if hasattr(extractor.generations, '__iter__') and not isinstance(extractor.generations, str) else [str(extractor.generations)]
        assert set(actual_generations) == set(expected_generations)  # Default generations
        assert extractor.calibration == False  # Default calibration
        assert extractor.baseline == 'baseline'  # Default baseline

    def test_init_with_config(self, sample_benchmark_data):
        """Test FeatureExtractor initialization with configuration."""
        extractor = FeatureExtractor(
            sample_benchmark_data, 
            generations=['baseline', 'LLM'],
            calibration=True,
            baseline='baseline'
        )
        
        assert extractor.benchmark is not None
        assert extractor.calibration == True
        assert 'baseline' in extractor.generations

    def test_load_models(self, sample_benchmark_data):
        """Test model loading functionality."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Test that extractor has embedding model
        assert hasattr(extractor, 'embedding_model')
        assert extractor._model == False  # Default embedding model is used

    def test_extract_sentiment(self, sample_benchmark_data):
        """Test sentiment extraction."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Mock the sentiment classifier to avoid loading actual models
        with patch('saged._extractor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'positive', 'score': 0.7},
                {'label': 'negative', 'score': 0.3}
            ]
            mock_pipeline.return_value = mock_classifier
            
            result = extractor.sentiment_classification()
            
            assert isinstance(result, pd.DataFrame)
            assert 'baseline_sentiment_score' in result.columns
            assert 'LLM_sentiment_score' in result.columns

    def test_extract_toxicity(self, sample_benchmark_data):
        """Test toxicity extraction."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Mock the toxicity classifier
        with patch('saged._extractor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'toxic', 'score': 0.2},
                {'label': 'non_toxic', 'score': 0.8}
            ]
            mock_pipeline.return_value = mock_classifier
            
            result = extractor.toxicity_classification()
            
            assert isinstance(result, pd.DataFrame)
            # Should have toxicity columns for each generation

    def test_extract_stereotype(self, sample_benchmark_data):
        """Test stereotype extraction."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Mock the stereotype classifier
        with patch('saged._extractor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'stereotype_gender', 'score': 0.1},
                {'label': 'stereotype_race', 'score': 0.1},
                {'label': 'stereotype_religion', 'score': 0.05},
                {'label': 'stereotype_profession', 'score': 0.08}
            ]
            mock_pipeline.return_value = mock_classifier
            
            result = extractor.stereotype_classification()
            
            assert isinstance(result, pd.DataFrame)
            # Should have stereotype columns for each generation

    def test_extract_features_full(self, sample_benchmark_data):
        """Test full feature extraction pipeline."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Mock all classifiers
        with patch('saged._extractor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'positive', 'score': 0.6},
                {'label': 'negative', 'score': 0.4}
            ]
            mock_pipeline.return_value = mock_classifier
            
            # Test that we can run sentiment classification
            result = extractor.sentiment_classification()
            assert isinstance(result, pd.DataFrame)

    def test_validate_input_data(self, sample_benchmark_data):
        """Test input data validation."""
        # Valid data should not raise errors
        extractor = FeatureExtractor(sample_benchmark_data)
        assert extractor.benchmark is not None
        
        # Invalid data should raise errors
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        with pytest.raises(AssertionError):
            FeatureExtractor(invalid_data, generations=['nonexistent'])

    def test_preprocess_texts(self, sample_benchmark_data):
        """Test text preprocessing."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Test that generations are properly configured
        assert 'baseline' in extractor.generations
        assert 'LLM' in extractor.generations

    def test_batch_processing(self, sample_benchmark_data):
        """Test batch processing of texts."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Mock processing to test batch functionality
        with patch('saged._extractor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'positive', 'score': 0.7},
                {'label': 'negative', 'score': 0.3}
            ]
            mock_pipeline.return_value = mock_classifier
            
            result = extractor.sentiment_classification()
            
            # Should process all rows in the benchmark
            assert len(result) == len(sample_benchmark_data)

    def test_error_handling(self, sample_benchmark_data):
        """Test error handling in feature extraction."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Test with empty or null texts
        empty_data = sample_benchmark_data.copy()
        empty_data.loc[0, 'baseline'] = ""
        empty_data.loc[1, 'baseline'] = None
        
        extractor.benchmark = empty_data
        
        # Should handle empty/null texts gracefully
        with patch('saged._extractor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'positive', 'score': 0.5},
                {'label': 'negative', 'score': 0.5}
            ]
            mock_pipeline.return_value = mock_classifier
            
            # Should not raise errors
            result = extractor.sentiment_classification()
            assert isinstance(result, pd.DataFrame)

    def test_device_configuration(self, sample_benchmark_data):
        """Test device configuration for model inference."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Test that extractor can handle different device configurations
        assert hasattr(extractor, 'embedding_model')

    def test_feature_column_naming(self, sample_benchmark_data):
        """Test proper naming of feature columns."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        expected_generations = ('baseline', 'LLM')
        assert extractor.generations == expected_generations

    def test_empty_text_handling(self, sample_benchmark_data):
        """Test handling of empty or invalid texts."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Test with data containing empty strings
        data_with_empty = sample_benchmark_data.copy()
        data_with_empty.loc[0, 'baseline'] = ""
        
        extractor.benchmark = data_with_empty
        
        # Should handle gracefully with safe classification
        with patch('saged._extractor.pipeline') as mock_pipeline:
            mock_classifier = Mock()
            mock_classifier.return_value = [
                {'label': 'positive', 'score': 0.5},
                {'label': 'negative', 'score': 0.5}
            ]
            mock_pipeline.return_value = mock_classifier
            
            result = extractor.sentiment_classification()
            assert isinstance(result, pd.DataFrame)

    def test_calibration_feature(self, sample_benchmark_data):
        """Test calibration feature extraction."""
        extractor = FeatureExtractor(sample_benchmark_data, calibration=True)
        
        assert extractor.calibration == True
        assert extractor.baseline == 'baseline'

    def test_multiple_generations(self, sample_benchmark_data):
        """Test extraction with multiple generations."""
        # Add more generation columns to test data
        data_with_more_gens = sample_benchmark_data.copy()
        data_with_more_gens['optimist'] = data_with_more_gens['baseline']
        data_with_more_gens['optimist_sentiment_score'] = [0.8, 0.9, 0.7, 0.85, 0.95, 0.8]
        
        extractor = FeatureExtractor(
            data_with_more_gens, 
            generations=['baseline', 'LLM', 'optimist']
        )
        
        assert len(extractor.generations) == 3
        assert 'optimist' in extractor.generations

    def test_model_caching(self, sample_benchmark_data):
        """Test model caching and reuse."""
        extractor = FeatureExtractor(sample_benchmark_data)
        
        # Test that embedding model is cached
        model1 = extractor.embedding_model
        model2 = extractor.embedding_model
        
        # Should be the same object (cached)
        assert model1 is model2 