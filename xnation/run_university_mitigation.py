import os
import json
import pandas as pd
from datetime import datetime
from xnation._mpf_pipeline import mpf_pipeline
from xnation.LLMFactory import LLMFactory

def create_generation_function(model_name="deepseek-r1-distill-qwen-1.5b", system_prompt="You are a helpful assistant."):
    """Create a generation function using LLMFactory with a specific system prompt."""
    llm = LLMFactory(model_name=model_name)
    
    def generation_function(text):
        try:
            # Create a simple prompt template for the generation
            prompt = f"""
            According to your system prompt, generate one comprehensive sentence to answer the following question:
            {text}
            """
            
            response = llm.client.chat.completions.create(
                model=llm.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generation: {e}")
            return None

    return generation_function


def main():
    # Source directory containing the existing data
    source_dir = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\universities\run_20250516_121354"
    
    # Load system prompts
    with open(os.path.join(source_dir, "system_prompts.json"), 'r') as f:
        system_prompts = json.load(f)
    
    # Load benchmark data
    benchmark_df = pd.read_csv(os.path.join(source_dir, "extractions.csv"))
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("data", "universities", f"mitigation_run_{timestamp}")
    
    # Create system-promptable generation function (required for 'sampled' and 'aggregated' modes)
    def system_promptable_generation_function(prompt, system_prompt):
        return create_generation_function(
            model_name="qwen-turbo-latest",
            system_prompt=system_prompt
        )(prompt)
    
    # Run the pipeline with modified parameters
    mpf_pipeline(
        benchmark_df=benchmark_df,
        system_prompts=system_prompts,
        output_path=output_dir,
        mitigation_type="wasserstein_weighted",
        alpha=0,  # L2 regularization strength
        beta=1,   # Sparsity penalty strength
        generation_mode="sampled_pre",  # Using pre-generated responses
        num_generations=3,  # Default number of generations
        save_interval=20,  # Default save interval
        baseline="baseline",  # Default baseline
        feature="sentiment",  # Default feature
        component_generations=['optimist', 'realist', 'empathetic', 'cautious', 'critical'],  # Explicitly list generations to use
        system_promptable_generation_function=system_promptable_generation_function  # Pass the system-promptable function
    )

if __name__ == "__main__":
    main() 