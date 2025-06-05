import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

# Import with fallback for direct execution
try:
    from ..models.experiment import Experiment
    from ..models.results import AnalysisResult
    from .saged_service import SAGEDService
except ImportError:
    from models.experiment import Experiment
    from models.results import AnalysisResult
    from saged_service import SAGEDService


class ExperimentService:
    """Service for managing experiment execution and background tasks"""
    
    def __init__(self):
        self.saged_service = SAGEDService()
        self.running_experiments = {}  # Track running experiments
    
    async def start_experiment(
        self,
        experiment: Experiment,
        db: Session,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Start experiment execution in background"""
        
        experiment_id = experiment.id
        
        if experiment_id in self.running_experiments:
            return {"success": False, "error": "Experiment already running"}
        
        # Mark experiment as running
        experiment.status = "running"
        experiment.started_at = datetime.utcnow()
        db.commit()
        
        self.running_experiments[experiment_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Starting experiment..."
        }
        
        try:
            # Get benchmark data path
            benchmark_data_path = self._get_benchmark_data_path(experiment.benchmark_id)
            
            # Run the SAGED pipeline
            result = await self.saged_service.run_experiment_pipeline(
                experiment_config=experiment.config,
                benchmark_data_path=benchmark_data_path,
                experiment_id=experiment_id,
                progress_callback=lambda progress, message: self._update_progress(
                    experiment_id, progress, message
                )
            )
            
            # Update experiment status
            if result["success"]:
                experiment.status = "completed"
                experiment.completed_at = datetime.utcnow()
                
                # Save analysis results
                if "results" in result:
                    await self._save_analysis_results(experiment, result["results"], db)
                
                self.running_experiments[experiment_id]["status"] = "completed"
                self.running_experiments[experiment_id]["progress"] = 1.0
                self.running_experiments[experiment_id]["message"] = "Experiment completed successfully"
                
            else:
                experiment.status = "failed"
                experiment.error_message = result.get("error", "Unknown error")
                
                self.running_experiments[experiment_id]["status"] = "failed"
                self.running_experiments[experiment_id]["error"] = experiment.error_message
            
            db.commit()
            
            return result
            
        except Exception as e:
            experiment.status = "failed"
            experiment.error_message = str(e)
            db.commit()
            
            if experiment_id in self.running_experiments:
                self.running_experiments[experiment_id]["status"] = "failed"
                self.running_experiments[experiment_id]["error"] = str(e)
            
            return {"success": False, "error": str(e)}
    
    def get_experiment_progress(self, experiment_id: int) -> Dict[str, Any]:
        """Get current progress of running experiment"""
        
        if experiment_id not in self.running_experiments:
            return {"status": "not_found"}
        
        return self.running_experiments[experiment_id]
    
    def stop_experiment(self, experiment_id: int, db: Session) -> Dict[str, Any]:
        """Stop running experiment"""
        
        if experiment_id not in self.running_experiments:
            return {"success": False, "error": "Experiment not running"}
        
        # Update database
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if experiment:
            experiment.status = "cancelled"
            db.commit()
        
        # Remove from running experiments
        del self.running_experiments[experiment_id]
        
        return {"success": True, "message": "Experiment stopped"}
    
    async def _update_progress(self, experiment_id: int, progress: float, message: str):
        """Update experiment progress"""
        if experiment_id in self.running_experiments:
            self.running_experiments[experiment_id]["progress"] = progress
            self.running_experiments[experiment_id]["message"] = message
    
    def _get_benchmark_data_path(self, benchmark_id: int) -> str:
        """Get path to benchmark data file"""
        return f"data/app_data/benchmarks/benchmark_{benchmark_id}_saged_data.json"
    
    async def _save_analysis_results(
        self,
        experiment: Experiment,
        results: Dict[str, Any],
        db: Session
    ):
        """Save analysis results to database"""
        
        # Create analysis result entry
        analysis_result = AnalysisResult(
            experiment_id=experiment.id,
            feature_results=results.get("disparity_results", []),
            summary_stats=results.get("summary", {}),
            created_at=datetime.utcnow()
        )
        
        db.add(analysis_result)
        db.commit()
    
    def cleanup_completed_experiments(self, max_completed: int = 10):
        """Clean up completed experiment tracking"""
        completed = [
            exp_id for exp_id, data in self.running_experiments.items()
            if data["status"] in ["completed", "failed", "cancelled"]
        ]
        
        if len(completed) > max_completed:
            # Remove oldest completed experiments
            for exp_id in completed[:-max_completed]:
                del self.running_experiments[exp_id] 