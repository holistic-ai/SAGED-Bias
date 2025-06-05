import json
import yaml
import requests
import backoff
from openai import AzureOpenAI
import http.client
import ollama

class ContentFormatter:
    @staticmethod
    def chat_completions(text, settings_params):
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
        data = {"messages": message, **settings_params}
        return json.dumps(data)

class AzureAgent:
    def __init__(self, model_name):
        with open("settings.yaml", "r") as stream:
            try:
                model_settings = yaml.safe_load(stream)[model_name]
            except yaml.YAMLError as exc:
                print(exc)
                return

        self.azure_uri = model_settings['AZURE_ENDPOINT_URL']
        self.headers = {
            'Authorization': f"Bearer {model_settings['AZURE_ENDPOINT_API_KEY']}",
            'Content-Type': 'application/json'
        }
        self.chat_formatter = ContentFormatter

    def invoke(self, text, **kwargs):
        body = self.chat_formatter.chat_completions(text, {**kwargs})
        conn = http.client.HTTPSConnection(self.azure_uri)
        conn.request("POST", '/v1/chat/completions', body=body, headers=self.headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        decoded_data = data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        content = parsed_data["choices"][0]["message"]["content"]
        return content

class GPTAgent:
    def __init__(self, model_name):
        with open("settings.yaml", "r") as stream:
            try:
                model_settings = yaml.safe_load(stream)[model_name]
            except yaml.YAMLError as exc:
                print(exc)
                return

        self.client = AzureOpenAI(
            api_key=model_settings['AZURE_OPENAI_KEY'],
            api_version=model_settings['AZURE_OPENAI_VERSION'],
            azure_endpoint=model_settings['AZURE_OPENAI_ENDPOINT']
        )
        self.deployment_name = model_settings['AZURE_DEPLOYMENT_NAME']

    def invoke(self, text, **kwargs):
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            **kwargs
        )
        return response.choices[0].message.content

class OllamaModel:
    def __init__(self,  base_model = 'llama3', system_prompt = 'You are a helpful assistant', model_name = 'llama3o', **kwargs):
        self.base_model = base_model
        self.model_name = model_name
        self.model_create(model_name, system_prompt, base_model, **kwargs)

    def model_create(self, model_name, system_prompt, base_model, **kwargs):
        modelfile = f'FROM {base_model}\nSYSTEM {system_prompt}\n'
        for key, value in kwargs.items():
            modelfile += f'PARAMETER {key.lower()} {value}\n'
        # print(modelfile)
        ollama.create(model=model_name, modelfile=modelfile)

    def invoke(self, prompt):
            answer = ollama.generate(model=self.model_name, prompt=prompt)
            return answer['response']

