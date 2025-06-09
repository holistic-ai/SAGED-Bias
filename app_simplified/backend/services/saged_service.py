from typing import Dict, Any, Optional
from saged import Pipeline
from ..schemas.build_config import DomainBenchmarkConfig, BenchmarkResponse
from .database_service import DatabaseService
from .model_service import ModelService
from sqlalchemy.exc import SQLAlchemyError
import logging
import sys
import io
from contextlib import contextmanager
from datetime import datetime
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SagedService')

# Silence wikipediaapi logs
logging.getLogger('wikipediaapi').setLevel(logging.WARNING)

class SagedService:
    def __init__(self, model_name: str = "qwen-turbo-latest"):
        logger.info("Initializing SagedService")
        self.pipeline = Pipeline()
        self.db_service = DatabaseService()
        self.model_service = ModelService(model_name=model_name)
        logger.info("SagedService initialized successfully")
    
    @contextmanager
    def _capture_pipeline_output(self):
        """
        Context manager to capture pipeline's stdout and convert it to logging.
        """
        # Create a string buffer to capture the output
        captured_output = io.StringIO()
        # Store the original stdout
        original_stdout = sys.stdout
        try:
            # Redirect stdout to our buffer
            sys.stdout = captured_output
            yield
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            # Get the captured output
            output = captured_output.getvalue()
            # Log each line of the output
            for line in output.splitlines():
                if line.strip():  # Only log non-empty lines
                    logger.info(f"[Pipeline] {line.strip()}")
    
    def _verify_database_connection(self):
        """
        Verify that the database is connected and activated.
        Raises an exception if the database is not properly configured.
        """
        try:
            logger.info("Starting database verification")
            
            # First activate the database
            logger.debug("Activating database")
            self.db_service.activate_database()
            
            # Check if database is connected
            logger.debug("Checking database connection")
            if not self.db_service.is_connected():
                logger.error("Database connection check failed")
                raise Exception("Database is not connected")
            
            # Check if database is activated
            logger.debug("Checking database activation status")
            if not self.db_service.is_activated():
                logger.error("Database activation check failed")
                raise Exception("Database is not activated")
            
            # Test a simple query to verify connection
            logger.debug("Testing database connection with query")
            self.db_service.test_connection()
            
            logger.info("Database verification completed successfully")
            
        except SQLAlchemyError as e:
            logger.error(f"Database SQL error during verification: {str(e)}")
            raise Exception(f"Database connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Database verification failed: {str(e)}")
            raise Exception(f"Database verification failed: {str(e)}")
    
    def _cleanup_database(self):
        """
        Cleanup database after operations are complete.
        """
        try:
            logger.info("Starting database cleanup")
            self.db_service.deactivate_database()
            logger.info("Database cleanup completed successfully")
        except Exception as e:
            logger.warning(f"Failed to deactivate database: {str(e)}")
            print(f"Warning: Failed to deactivate database: {str(e)}")
    
    async def build_benchmark(self, domain: str, config: DomainBenchmarkConfig) -> BenchmarkResponse:
        """
        Build a benchmark using the SAGED pipeline with database storage.
        
        Args:
            domain: The domain for the benchmark
            config: The benchmark configuration
            
        Returns:
            BenchmarkResponse: The response containing the benchmark results
        """
        logger.info(f"Starting benchmark build for domain: {domain}")
        try:
            # Verify database connection before proceeding
            self._verify_database_connection()
            
            # Convert Pydantic model to dict for the pipeline
            logger.debug("Converting configuration to dictionary")
            config_dict = config.model_dump()
            
            # Add database configuration
            logger.debug("Adding database configuration")
            config_dict['use_database'] = True
            config_dict['database_config'] = self.db_service.get_database_config()
            
            # Update saving locations to use database table names
            logger.debug("Updating saving locations for database tables")
            if 'shared_config' not in config_dict:
                config_dict['shared_config'] = {}
            
            shared_config = config_dict['shared_config']
            
            # Get all table names once
            keywords_table = self.db_service.get_table_name('keywords', domain)
            source_finder_table = self.db_service.get_table_name('source_finder', domain)
            scraped_sentences_table = self.db_service.get_table_name('scraped_sentences', domain)
            benchmark_table = self.db_service.get_table_name('benchmark', domain)
            replacement_description_table = self.db_service.get_table_name('replacement_description', domain) if config_dict.get('branching', False) else None
            
            # Update keyword finder config to use database table name
            if 'keyword_finder' not in shared_config:
                shared_config['keyword_finder'] = {}
            shared_config['keyword_finder']['saving_location'] = keywords_table
            
            # Update source finder config to use database table name
            if 'source_finder' not in shared_config:
                shared_config['source_finder'] = {}
            shared_config['source_finder']['saving_location'] = source_finder_table
            
            # Update scraper config to use database table name
            if 'scraper' not in shared_config:
                shared_config['scraper'] = {}
            shared_config['scraper']['saving_location'] = scraped_sentences_table
            
            # Update prompt assembler config to use database table name
            if 'prompt_assembler' not in shared_config:
                shared_config['prompt_assembler'] = {}
            shared_config['prompt_assembler']['saving_location'] = benchmark_table
            
            # Update main saving location to use database table name
            config_dict['saving_location'] = benchmark_table
            
            # Handle branching configuration if enabled
            if config_dict.get('branching', False):
                logger.debug("Processing branching configuration")
                if 'branching_config' not in config_dict:
                    config_dict['branching_config'] = {}
                
                # Update replacement description saving location for branching
                if 'replacement_description_saving_location' in config_dict['branching_config']:
                    config_dict['branching_config']['replacement_description_saving_location'] = replacement_description_table
            
            # Check if we need to use question generation
            generation_function = None
            model_info = None
            if 'prompt_assembler' in shared_config and shared_config['prompt_assembler'].get('method') == 'questions':
                logger.info("Question generation method detected, creating generation function")
                # Create generation function with default model and system prompt
                generation_function = self.model_service.create_generation_function()
                # Store model information for metadata
                model_info = {
                    'model_name': self.model_service.model_name,
                    'is_azure': self.model_service.is_azure,
                    'deployment_name': self.model_service.model_name if self.model_service.is_azure else None
                }
                # Add the generation function to the prompt assembler config
                shared_config['prompt_assembler']['generation_function'] = generation_function
            
            # Print the config dictionary
            logger.info("Config dictionary:")
            logger.info(config_dict)

            # Build the benchmark with output capture
            logger.info("Starting SAGED pipeline benchmark build")
            with self._capture_pipeline_output():
                benchmark_result = self.pipeline.build_benchmark(domain=domain, config=config_dict)
            logger.info("SAGED pipeline benchmark build completed")
            
            # Create a copy of config_dict for metadata storage
            metadata_config = copy.deepcopy(config_dict)
            # Replace generation function with model info in the metadata config
            if generation_function and 'prompt_assembler' in metadata_config['shared_config']:
                metadata_config['shared_config']['prompt_assembler'].pop('generation_function', None)
                if model_info:
                    metadata_config['shared_config']['prompt_assembler']['generation_function'] = model_info
            
            # Get the data from the benchmark result
            logger.debug("Processing benchmark results")
            result_dict = {
                "domain": domain,
                "data": benchmark_result.data.to_dict() if hasattr(benchmark_result, 'data') else None,
                "table_names": {
                    "keywords": keywords_table,
                    "source_finder": source_finder_table,
                    "scraped_sentences": scraped_sentences_table,
                    "benchmark": benchmark_table,
                    "replacement_description": replacement_description_table
                },
                "configuration": metadata_config,  # Use the cleaned config for metadata
                "database_config": self.db_service.get_database_config(),
                "time_stamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save the benchmark metadata to our database
            logger.info("Saving benchmark metadata to database")
            metadata_table_name = f"metadata_benchmark_{domain}_{benchmark_table}"
            self.db_service.save_benchmark_metadata(metadata_table_name, result_dict)
            
            # Cleanup database after successful operation
            self._cleanup_database()
            
            logger.info(f"Benchmark build completed successfully for domain: {domain}")
            return BenchmarkResponse(
                status="success",
                message="Benchmark built successfully",
                data=result_dict
            )
            
        except Exception as e:
            logger.error(f"Error during benchmark build: {str(e)}")
            # Cleanup database even if there's an error
            self._cleanup_database()
            return BenchmarkResponse(
                status="error",
                message=f"Failed to build benchmark: {str(e)}",
                data=None
            )

    async def get_benchmark_status(self, domain: str) -> BenchmarkResponse:
        """
        Get the status of a benchmark for a given domain.
        
        Args:
            domain: The domain to check
            
        Returns:
            BenchmarkResponse: The response containing the benchmark status
        """
        logger.info(f"Getting benchmark status for domain: {domain}")
        try:
            # Verify database connection before proceeding
            self._verify_database_connection()
            
            # Get the latest benchmark data
            logger.debug("Retrieving latest benchmark data")
            data = self.db_service.get_latest_benchmark(domain)
            
            # Cleanup database after successful operation
            self._cleanup_database()
            
            status = "completed" if data else "not_found"
            logger.info(f"Benchmark status retrieved: {status}")
            
            return BenchmarkResponse(
                status="success",
                message=f"Benchmark status retrieved for domain: {domain}",
                data={
                    "status": status,
                    "data": data
                }
            )
        except Exception as e:
            logger.error(f"Error getting benchmark status: {str(e)}")
            # Cleanup database even if there's an error
            self._cleanup_database()
            return BenchmarkResponse(
                status="error",
                message=f"Failed to get benchmark status: {str(e)}",
                data=None
            ) 