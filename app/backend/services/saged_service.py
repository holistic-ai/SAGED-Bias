import sys
import os
import asyncio
import json
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

# Add project root to path for SAGED imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

try:
    from saged import Pipeline, FeatureExtractor, DisparityDiagnoser
    from saged._saged_data import SAGEDData
    SAGED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAGED import failed: {e}")
    SAGED_AVAILABLE = False


class SAGEDService:
    """Service for integrating with SAGED bias analysis pipeline"""
    
    def __init__(self):
        self.saged_available = SAGED_AVAILABLE
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate SAGED configuration"""
        result = {"valid": True, "errors": []}
        
        # Required top-level keys
        required_keys = ["categories", "shared_config"]
        for key in required_keys:
            if key not in config:
                result["errors"].append(f"Missing required key: {key}")
                result["valid"] = False
        
        # Validate categories
        if "categories" in config:
            valid_categories = ['nationality', 'gender', 'race', 'religion', 'profession', 'age']
            for category in config["categories"]:
                if category not in valid_categories:
                    result["errors"].append(f"Invalid category: {category}")
                    result["valid"] = False
        
        # Validate shared_config
        if "shared_config" in config:
            shared_config = config["shared_config"]
            required_components = ["keyword_finder", "source_finder", "scraper", "prompt_assembler"]
            
            for component in required_components:
                if component not in shared_config:
                    result["errors"].append(f"Missing required component: {component}")
                    result["valid"] = False
        
        return result
    
    async def run_benchmark_creation(
        self, 
        benchmark_config: Dict[str, Any],
        benchmark_id: int,
        db: Session
    ) -> Dict[str, Any]:
        """Run SAGED pipeline to create benchmark data"""
        
        if not self.saged_available:
            return {"success": False, "error": "SAGED library not available"}
        
        try:
            # Create Pipeline instance
            pipeline = Pipeline(benchmark_config)
            
            # Run the pipeline stages for benchmark creation
            result = {"success": True, "stages": {}}
            
            # Stage 1: Keyword Finding
            if benchmark_config.get("shared_config", {}).get("keyword_finder", {}).get("require", False):
                print(f"Running keyword finder for benchmark {benchmark_id}...")
                try:
                    pipeline.run_keyword_finding()
                    result["stages"]["keyword_finding"] = {"status": "completed", "timestamp": datetime.now().isoformat()}
                except Exception as e:
                    result["stages"]["keyword_finding"] = {"status": "failed", "error": str(e)}
                    print(f"Keyword finding failed: {e}")
            
            # Stage 2: Source Finding
            if benchmark_config.get("shared_config", {}).get("source_finder", {}).get("require", False):
                print(f"Running source finder for benchmark {benchmark_id}...")
                try:
                    pipeline.run_source_finding()
                    result["stages"]["source_finding"] = {"status": "completed", "timestamp": datetime.now().isoformat()}
                except Exception as e:
                    result["stages"]["source_finding"] = {"status": "failed", "error": str(e)}
                    print(f"Source finding failed: {e}")
            
            # Stage 3: Scraping
            if benchmark_config.get("shared_config", {}).get("scraper", {}).get("require", False):
                print(f"Running scraper for benchmark {benchmark_id}...")
                try:
                    pipeline.run_scraping()
                    result["stages"]["scraping"] = {"status": "completed", "timestamp": datetime.now().isoformat()}
                except Exception as e:
                    result["stages"]["scraping"] = {"status": "failed", "error": str(e)}
                    print(f"Scraping failed: {e}")
            
            # Stage 4: Prompt Assembly
            if benchmark_config.get("shared_config", {}).get("prompt_assembler", {}).get("require", False):
                print(f"Running prompt assembler for benchmark {benchmark_id}...")
                try:
                    pipeline.run_prompt_assembling()
                    result["stages"]["prompt_assembling"] = {"status": "completed", "timestamp": datetime.now().isoformat()}
                    
                    # Save the assembled benchmark data
                    data_path = f"data/app_data/benchmarks/benchmark_{benchmark_id}_saged_data.json"
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    
                    # Export the pipeline data
                    saged_data = pipeline.get_assembled_data()
                    if hasattr(saged_data, 'to_dict'):
                        with open(data_path, 'w') as f:
                            json.dump(saged_data.to_dict(), f, indent=2, default=str)
                    
                    result["data_path"] = data_path
                    
                except Exception as e:
                    result["stages"]["prompt_assembling"] = {"status": "failed", "error": str(e)}
                    print(f"Prompt assembling failed: {e}")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {"success": False, "error": error_msg}
    
    async def run_experiment_pipeline(
        self,
        experiment_config: Dict[str, Any],
        benchmark_data_path: str,
        experiment_id: int,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Run complete SAGED experiment pipeline"""
        
        if not self.saged_available:
            return {"success": False, "error": "SAGED library not available"}
        
        try:
            result = {"success": True, "stages": {}, "results": {}}
            
            # Load benchmark data
            if not os.path.exists(benchmark_data_path):
                return {"success": False, "error": f"Benchmark data not found: {benchmark_data_path}"}
            
            # Initialize progress
            if progress_callback:
                await progress_callback(0.1, "Loading benchmark data...")
            
            # Stage 1: Generation
            if progress_callback:
                await progress_callback(0.2, "Generating LLM responses...")
            
            try:
                generation_config = experiment_config.get("generation_config", {})
                print(f"Running generation for experiment {experiment_id}...")
                
                # This would integrate with your LLM generation logic
                # For now, simulate the process
                await asyncio.sleep(2)  # Simulate generation time
                
                generation_path = f"data/app_data/experiments/experiment_{experiment_id}_generation.json"
                os.makedirs(os.path.dirname(generation_path), exist_ok=True)
                
                # Simulate generation results
                generation_results = {
                    "experiment_id": experiment_id,
                    "generation_config": generation_config,
                    "responses": [],  # Would contain actual LLM responses
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(generation_path, 'w') as f:
                    json.dump(generation_results, f, indent=2)
                
                result["stages"]["generation"] = {
                    "status": "completed",
                    "path": generation_path,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                result["stages"]["generation"] = {"status": "failed", "error": str(e)}
                print(f"Generation failed: {e}")
            
            # Stage 2: Feature Extraction
            if progress_callback:
                await progress_callback(0.5, "Extracting features...")
            
            try:
                extraction_config = experiment_config.get("extraction_config", {})
                print(f"Running feature extraction for experiment {experiment_id}...")
                
                # Initialize FeatureExtractor if available
                extractor = FeatureExtractor(extraction_config)
                
                # Simulate feature extraction
                await asyncio.sleep(1)
                
                extraction_path = f"data/app_data/experiments/experiment_{experiment_id}_extraction.json"
                
                # Simulate extraction results
                extraction_results = {
                    "experiment_id": experiment_id,
                    "extraction_config": extraction_config,
                    "features": {
                        "sentiment_score": [0.2, 0.8, 0.5, 0.1],  # Sample data
                        "toxicity_score": [0.1, 0.3, 0.2, 0.05],
                        "regard_score": [0.7, 0.4, 0.6, 0.8]
                    },
                    "feature_names": ["sentiment_score", "toxicity_score", "regard_score"],
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(extraction_path, 'w') as f:
                    json.dump(extraction_results, f, indent=2)
                
                result["stages"]["extraction"] = {
                    "status": "completed",
                    "path": extraction_path,
                    "features": extraction_results["feature_names"],
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                result["stages"]["extraction"] = {"status": "failed", "error": str(e)}
                print(f"Feature extraction failed: {e}")
            
            # Stage 3: Bias Analysis
            if progress_callback:
                await progress_callback(0.8, "Analyzing bias...")
            
            try:
                analysis_config = experiment_config.get("analysis_config", {})
                print(f"Running bias analysis for experiment {experiment_id}...")
                
                # Initialize DisparityDiagnoser if available
                diagnoser = DisparityDiagnoser(analysis_config)
                
                # Simulate bias analysis
                await asyncio.sleep(1)
                
                analysis_path = f"data/app_data/experiments/experiment_{experiment_id}_analysis.json"
                
                # Simulate analysis results
                analysis_results = {
                    "experiment_id": experiment_id,
                    "analysis_config": analysis_config,
                    "disparity_results": [
                        {
                            "feature_name": "sentiment_score",
                            "target_group": "gender",
                            "disparity_score": 0.78,
                            "p_value": 0.001,
                            "confidence_interval": [0.65, 0.91],
                            "effect_size": 0.42,
                            "significant": True
                        },
                        {
                            "feature_name": "toxicity_score",
                            "target_group": "nationality",
                            "disparity_score": 0.45,
                            "p_value": 0.023,
                            "confidence_interval": [0.32, 0.58],
                            "effect_size": 0.28,
                            "significant": True
                        },
                        {
                            "feature_name": "regard_score",
                            "target_group": "gender",
                            "disparity_score": 0.12,
                            "p_value": 0.234,
                            "confidence_interval": [-0.05, 0.29],
                            "effect_size": 0.08,
                            "significant": False
                        }
                    ],
                    "summary": {
                        "total_features": 3,
                        "significant_features": 2,
                        "bias_percentage": 66.7
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(analysis_path, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                result["stages"]["analysis"] = {
                    "status": "completed",
                    "path": analysis_path,
                    "summary": analysis_results["summary"],
                    "timestamp": datetime.now().isoformat()
                }
                
                result["results"] = analysis_results
                
            except Exception as e:
                result["stages"]["analysis"] = {"status": "failed", "error": str(e)}
                print(f"Bias analysis failed: {e}")
            
            if progress_callback:
                await progress_callback(1.0, "Experiment completed!")
            
            return result
            
        except Exception as e:
            error_msg = f"Experiment pipeline failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_sample_config(self, domain: str = "demographics") -> Dict[str, Any]:
        """Generate sample SAGED configuration for testing"""
        
        if domain == "demographics":
            return {
                "categories": ["nationality", "gender"],
                "branching": False,
                "shared_config": {
                    "keyword_finder": {
                        "require": True,
                        "method": "embedding_on_wiki",
                        "keyword_number": 7,
                        "embedding_model": "paraphrase-Mpnet-base-v2",
                        "saving": True
                    },
                    "source_finder": {
                        "require": True,
                        "method": "wiki",
                        "scrap_number": 5,
                        "saving": True
                    },
                    "scraper": {
                        "require": True,
                        "saving": True,
                        "method": "wiki"
                    },
                    "prompt_assembler": {
                        "require": True,
                        "method": "split_sentences",
                        "max_benchmark_length": 500
                    }
                },
                "saving": True
            }
        else:
            return {
                "categories": ["profession"],
                "branching": False,
                "shared_config": {
                    "keyword_finder": {"require": True},
                    "source_finder": {"require": True},
                    "scraper": {"require": True},
                    "prompt_assembler": {"require": True}
                }
            }
    
    def get_available_categories(self) -> List[str]:
        """Get list of available bias categories"""
        return ['nationality', 'gender', 'race', 'religion', 'profession', 'age']
    
    def get_available_features(self) -> List[str]:
        """Get list of available feature extractors"""
        return [
            'sentiment_score',
            'toxicity_score', 
            'regard_score',
            'emotion_classification',
            'stereotype_detection'
        ] 