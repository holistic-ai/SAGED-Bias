from typing import Dict, Any, Optional, Callable
from saged.mpf.LLMFactory import LLMFactory
import logging
import os
from pathlib import Path
from openai import AzureOpenAI, OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelService')

class ModelService:
    def __init__(self, model_name: str = "qwen-turbo-latest"):
        """
        Initialize the ModelService with a specific model.
        
        Args:
            model_name (str): Name of the model to use (default: "qwen-turbo-latest")
        """
        logger.info(f"Initializing ModelService with model: {model_name}")
        try:
            # Get the project root directory
            project_root = Path(__file__).parent.parent.parent.parent
            
            # Load settings
            import yaml
            settings_path = project_root / "settings.yaml"
            with open(settings_path, 'r') as f:
                self.settings = yaml.safe_load(f)
            
            # Initialize appropriate client based on model type
            if model_name in self.settings:
                model_config = self.settings[model_name]
                if 'AZURE_OPENAI_KEY' in model_config:
                    # Initialize Azure OpenAI client
                    self.client = AzureOpenAI(
                        api_key=model_config['AZURE_OPENAI_KEY'],
                        api_version=model_config['AZURE_OPENAI_VERSION'],
                        azure_endpoint=model_config['AZURE_OPENAI_ENDPOINT']
                    )
                    self.model_name = model_config['AZURE_DEPLOYMENT_NAME']
                    self.is_azure = True
                else:
                    # Initialize DashScope client
                    self.client = OpenAI(
                        api_key=model_config['DASHSCOPE_API_KEY'],
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )
                    self.model_name = model_name
                    self.is_azure = False
            else:
                raise ValueError(f"Model '{model_name}' not found in settings")
            
            logger.info("ModelService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ModelService: {str(e)}")
            raise
    
    def run_prompt(self, prompt_template_name: str, **kwargs) -> str:
        """
        Run a prompt through the LLM using a template.
        
        Args:
            prompt_template_name (str): Name of the prompt template file
            **kwargs: Variables to substitute in the template
            
        Returns:
            str: The LLM response
        """
        try:
            logger.info(f"Running prompt template: {prompt_template_name}")
            # Load prompt template
            project_root = Path(__file__).parent.parent.parent.parent
            prompt_template_path = project_root / "saged" / "mpf" / "prompts" / f"{prompt_template_name}.txt"
            
            with open(prompt_template_path, 'r', encoding="utf-8") as f:
                from string import Template
                template = Template(f.read())
                prompt = template.safe_substitute(**kwargs)
            
            # Run the prompt through the appropriate client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a linguistic analysis system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            logger.info("Prompt executed successfully")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to run prompt: {str(e)}")
            raise
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get information about available models from settings.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information
        """
        try:
            # Return the models from settings, excluding sensitive information
            models = {}
            for model_name, config in self.settings.items():
                models[model_name] = {
                    'type': 'azure' if 'AZURE_OPENAI_KEY' in config else 'dashscope',
                    'deployment_name': config.get('AZURE_DEPLOYMENT_NAME', model_name)
                }
            
            return {
                "status": "success",
                "models": models
            }
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "models": {}
            }
    
    def create_generation_function(self, model_name: str = None, system_prompt: str = "You are a helpful assistant.") -> Callable:
        """
        Create a generation function for question generation.
        
        Args:
            model_name (str, optional): Name of the model to use. If None, uses the default model.
            system_prompt (str): System prompt for the model.
            
        Returns:
            Callable: A function that takes text input and returns generated text.
        """
        try:
            logger.info(f"Creating generation function with model: {model_name or self.model_name}")
            
            # Use the specified model or default to the current one
            if model_name is not None and model_name != self.model_name:
                # Create a new instance with the specified model
                model_service = ModelService(model_name=model_name)
                client = model_service.client
                model = model_service.model_name
            else:
                client = self.client
                model = self.model_name
            
            def generation_function(text: str) -> Optional[str]:
                try:
                    # Create a simple prompt template for the generation
                    prompt = f"""
                    According to your system prompt, generate one comprehensive sentence to answer the following question:
                    {text}
                    """
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error in generation: {str(e)}")
                    return None
            
            logger.info("Generation function created successfully")
            return generation_function
            
        except Exception as e:
            logger.error(f"Failed to create generation function: {str(e)}")
            raise 