import os
import json
import pandas as pd
from datetime import datetime
from ._mpf_pipeline import mpf_pipeline
from .LLMFactory import LLMFactory

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
    source_dir = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\full_universities\mitigation_run_20250516_212213"
    
    # Load system prompts
    with open(os.path.join(source_dir, "system_prompts.json"), 'r') as f:
        system_prompts = json.load(f)
    
    # Load benchmark data
    benchmark_df = pd.read_csv(os.path.join(source_dir, "responses_with_sentiment_20250516_212213.csv"))
    
    # Create system-promptable generation function
    def system_promptable_generation_function(prompt, system_prompt):
        return create_generation_function(
            model_name="qwen-turbo-latest",
            system_prompt=system_prompt
        )(prompt)
    
    # Define parameter combinations
    mitigation_types = [
        'mixed_weighted',
        'calibration_weighted',
        'mean_weighted',
        'tv_weighted',
        'kl_weighted',
        'wasserstein_weighted',
    ]
    alphas = [0, 0.5]
    betas = [0, 3, 0.1, 1, 0.3]
    
    # Define metric weights for mixed objective
    metric_weights = {
        'kl': 0.2,        # Higher weight for KL divergence
        'calibration': 0.8  # Lower weight for calibration
    }
    
    # Define all metrics to use in reports
    report_metrics = [
        'wasserstein',
        'kl',
        'tv',
        'mean',
        'calibration'
    ]
    
    # Run experiments for each combination
    for mitigation_type in mitigation_types:
        for alpha in alphas:
            for beta in betas:
                # Create output directory with timestamp and parameters
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(
                    "data", 
                    "full_universities", 
                    f"mitigation_run_{mitigation_type}_a{alpha}_b{beta}_{timestamp}"
                )
                
                print(f"\nRunning experiment with:")
                print(f"Mitigation type: {mitigation_type}")
                print(f"Alpha: {alpha}")
                print(f"Beta: {beta}")
                if mitigation_type == 'mixed_weighted':
                    print(f"Metric weights: {metric_weights}")
                print(f"Report metrics: {', '.join(report_metrics)}")
                
                # Run the pipeline with current parameters
                mpf_pipeline(
                    benchmark_df=benchmark_df,
                    system_prompts=system_prompts,
                    output_path=output_dir,
                    mitigation_type=mitigation_type,
                    alpha=alpha,
                    beta=beta,
                    generation_mode="sampled_pre",
                    num_generations=3,
                    save_interval=20,
                    baseline="baseline",
                    feature="sentiment",
                    component_generations=['optimist', 'realist', 'empathetic', 'cautious', 'critical'],
                    system_promptable_generation_function=system_promptable_generation_function,
                    report_metric=report_metrics,  # Pass the list directly
                    metric_weights=metric_weights if mitigation_type == 'mixed_weighted' else None  # Add metric weights for mixed objective
                )

if __name__ == "__main__":
    main() 