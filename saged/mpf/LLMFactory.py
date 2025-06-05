import yaml
import os
import json
from string import Template
from openai import OpenAI
from .constantpath import CURRENT_DIR, PROJECT_ROOT


class LLMFactory:
    """
    Factory class to create LLM instances based on model name and configuration
    """

    def __init__(self, model_name = "deepseek-v3", settings_path=os.path.join(PROJECT_ROOT, "settings.yaml")):
        """
        Initialize LLM with specified model from settings

        Args:
            model_name (str): Name of the model to use
            settings_path (str): Path to the settings YAML file
        """
        self.model_name = model_name

        # Load settings
        with open(settings_path, 'r') as f:
            self.settings = yaml.safe_load(f)

        # Validate model exists in settings
        if model_name not in self.settings:
            raise ValueError(f"Model '{model_name}' not found in settings")

        # Get API key for the specified model
        self.api_key = self.settings[model_name].get('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError(f"API key not found for model '{model_name}'")

        self.base_url = self.settings.get('BASE_URL', "https://dashscope.aliyuncs.com/compatible-mode/v1")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def run_prompt(self, prompt_template_name, **kwargs):
        """
        Run a prompt through the LLM using a template with safe variable substitution

        Args:
            prompt_template_name (str): name of the prompt template file
            **kwargs: Variables to substitute in the template using $variable_name format

        Returns:
            str: The LLM response, sanitized for CSV storage
        """
        # Load prompt template
        prompt_template_path = os.path.join(CURRENT_DIR, "prompts", f"{prompt_template_name}.txt")
        with open(prompt_template_path, 'r', encoding="utf-8") as f:
            template = Template(f.read())

        # Replace variables in template using safe substitution
        try:
            prompt = template.safe_substitute(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable in prompt template: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid template variable: {e}")

        # Run the prompt through the LLM
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": "You are a linguistic analysis system."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "text"},
            temperature=0
        )

        # Get the raw response content and sanitize for CSV storage
        raw_response = response.choices[0].message.content
        # Replace newlines and carriage returns with spaces
        sanitized_response = raw_response.replace('\n', ' ').replace('\r', ' ')
        # Remove any double spaces created by the replacement
        sanitized_response = ' '.join(sanitized_response.split())
        # Handle quotes and commas to avoid CSV parsing issues
        sanitized_response = sanitized_response.replace('"', "'").replace('",', ',').replace(',"', ',')
        return sanitized_response