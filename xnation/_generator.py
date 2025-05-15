from saged._utility import check_benchmark, check_generation_function
import pandas as pd
import os
from tqdm import tqdm
import json

class ResponseGenerator:
    def __init__(self, benchmark):
        check_benchmark(benchmark)
        self.benchmark = benchmark

    @classmethod
    def prompt_template(cls, task):
        if task == 'completion':
            return lambda x: f'Compete the following one sentence to make it more meaningful: "{x}"'
        if task == 'question_answering':
            return lambda x: f'Answer the following question in one sentence: "{x}"'

    def generate(self,
                 generation_function,
                 generation_name='LLM',
                 remove_prompt = False,
                 add_prompt_to_generation = False,
                 max_generation_length=2000,
                 save_interval=20,
                 save_path=None):

        check_generation_function(generation_function)
        generation = generation_function
        print('Generating.....')
        
        # Create a temporary column for tracking progress
        self.benchmark[generation_name] = None
        
        # Process in batches
        total_rows = len(self.benchmark)
        for i in tqdm(range(0, total_rows, save_interval), desc="Overall Progress"):
            end_idx = min(i + save_interval, total_rows)
            print(f'Processing rows {i} to {end_idx}...')
            
            # Generate for current batch using iloc for integer-based indexing
            current_slice = self.benchmark.iloc[i:end_idx].copy()  # Make an explicit copy
            
            # Apply generation function with tqdm
            tqdm.pandas(desc="Batch Processing", leave=False)
            current_slice.loc[:, generation_name] = current_slice['prompts'].progress_apply(generation)
            
            # Apply length limit
            current_slice.loc[:, generation_name] = current_slice.apply(
                lambda x: x[generation_name][:max_generation_length] if pd.notna(x[generation_name]) else None, 
                axis=1
            )
            
            # Update the main dataframe with the processed slice
            self.benchmark.iloc[i:end_idx] = current_slice
            
            # Save progress if path is provided
            if save_path:
                temp_save_path = save_path
                self.benchmark.to_csv(temp_save_path, index=False)
                print(f'Progress saved to {temp_save_path}')

        # Apply final transformations
        if add_prompt_to_generation:
            self.benchmark[generation_name] = self.benchmark.apply(
                lambda x: x['prompts'] + x[generation_name] if pd.notna(x[generation_name]) else None,
                axis=1
            )
            
        if remove_prompt:
            self.benchmark['baseline'] = self.benchmark.apply(
                lambda x: x['baseline'].replace(x['prompts'], '') if pd.notna(x['baseline']) else None,
                axis=1
            )
            self.benchmark[generation_name] = self.benchmark.apply(
                lambda x: x[generation_name].replace(x['prompts'], '') if pd.notna(x[generation_name]) else None,
                axis=1
            )

        # Save final results if path is provided
        if save_path:
            self.benchmark.to_csv(save_path, index=False)

        return self.benchmark

    def routed_generate(self,
                       system_promptable_generation_function,
                       weights_json_path,
                       generation_name='LLM',
                       max_generation_length=2000,
                       save_interval=20,
                       save_path=None):
        """
        Generate responses using perspective-based routing based on optimized weights.
        For each prompt, selects a single perspective based on weights as probabilities
        and uses that perspective's system prompt for generation.
        """
        # Load the weights configuration
        with open(weights_json_path, 'r') as f:
            weights_config = json.load(f)
        
        # Create a temporary column for tracking progress
        self.benchmark[generation_name] = None
        
        # Process in batches
        total_rows = len(self.benchmark)
        for i in tqdm(range(0, total_rows, save_interval), desc="Overall Progress"):
            end_idx = min(i + save_interval, total_rows)
            
            # Generate for current batch
            current_slice = self.benchmark.iloc[i:end_idx].copy()
            
            # Apply generation function with perspective routing
            tqdm.pandas(desc="Batch Processing", leave=False)
            
            def apply_perspective_generation(row):
                concept = row['concept']
                if not concept:
                    return None
                
                # Get weights for this concept across all perspectives
                perspective_weights = {}
                total_weight = 0
                for perspective, config in weights_config['perspectives'].items():
                    weight = config['weights'].get(concept, 0)
                    if weight > 0:
                        perspective_weights[perspective] = weight
                        total_weight += weight
                
                if not perspective_weights:
                    return None
                
                # Select a perspective based on probabilities
                import random
                selected_perspective = random.choices(
                    list(perspective_weights.keys()),
                    weights=list(perspective_weights.values()),
                    k=1
                )[0]
                
                # Generate response using the prompt and selected perspective's system prompt
                return system_promptable_generation_function(
                    prompt=row['prompts'],
                    system_prompt=weights_config['perspectives'][selected_perspective]['system_prompt']
                )
            
            current_slice.loc[:, generation_name] = current_slice.progress_apply(apply_perspective_generation, axis=1)
            
            # Apply length limit
            current_slice.loc[:, generation_name] = current_slice.apply(
                lambda x: x[generation_name][:max_generation_length] if pd.notna(x[generation_name]) else None, 
                axis=1
            )
            
            # Update the main dataframe with the processed slice
            self.benchmark.iloc[i:end_idx] = current_slice
            
            # Save progress if path is provided
            if save_path:
                self.benchmark.to_csv(save_path, index=False)

        return self.benchmark

    def ensembled_generate(self,
                          system_promptable_generation_function,
                          weights_json_path,
                          num_generations=3,
                          ensemble_system_prompt="You are an expert at synthesizing multiple perspectives into a coherent response. Given multiple responses to the same prompt, create a balanced and comprehensive answer that captures the key points from all perspectives.",
                          generation_name='LLM',
                          max_generation_length=2000,
                          save_interval=20,
                          save_path=None):
        """
        Generate responses using an ensemble approach based on perspective weights.
        For each prompt:
        1. Generates n responses using different perspective selections
        2. Combines these responses into a template
        3. Uses a final LLM call to synthesize these responses into a coherent answer
        """
        # Load the weights configuration
        with open(weights_json_path, 'r') as f:
            weights_config = json.load(f)
        
        # Create a temporary column for tracking progress
        self.benchmark[generation_name] = None
        
        # Process in batches
        total_rows = len(self.benchmark)
        for i in tqdm(range(0, total_rows, save_interval), desc="Overall Progress"):
            end_idx = min(i + save_interval, total_rows)
            
            # Generate for current batch
            current_slice = self.benchmark.iloc[i:end_idx].copy()
            
            # Apply generation function with perspective routing
            tqdm.pandas(desc="Batch Processing", leave=False)
            
            def apply_ensemble_generation(row):
                concept = row['concept']
                if not concept:
                    return None
                
                # Get weights for this concept across all perspectives
                perspective_weights = {}
                total_weight = 0
                for perspective, config in weights_config['perspectives'].items():
                    weight = config['weights'].get(concept, 0)
                    if weight > 0:
                        perspective_weights[perspective] = weight
                        total_weight += weight
                
                if not perspective_weights:
                    return None
                
                # Generate n responses using different perspective selections
                import random
                generated_responses = []
                for _ in range(num_generations):
                    # Select a perspective based on probabilities
                    selected_perspective = random.choices(
                        list(perspective_weights.keys()),
                        weights=list(perspective_weights.values()),
                        k=1
                    )[0]
                    
                    # Generate response using the prompt and selected perspective's system prompt
                    response = system_promptable_generation_function(
                        prompt=row['prompts'],
                        system_prompt=weights_config['perspectives'][selected_perspective]['system_prompt']
                    )
                    generated_responses.append(response)
                
                # Create a template combining all responses
                ensemble_prompt = f"""Given the following {num_generations} different responses to the same prompt, create an middle answer from all perspectives:

Responses:
{chr(10).join([f'Response {i+1}: {resp}' for i, resp in enumerate(generated_responses)])}

Please synthesize these responses into a single coherent answer:"""
                
                # Generate final response using the ensemble system prompt
                final_response = system_promptable_generation_function(
                    prompt=ensemble_prompt,
                    system_prompt=ensemble_system_prompt
                )
                
                return final_response
            
            current_slice.loc[:, generation_name] = current_slice.progress_apply(apply_ensemble_generation, axis=1)
            
            # Apply length limit
            current_slice.loc[:, generation_name] = current_slice.apply(
                lambda x: x[generation_name][:max_generation_length] if pd.notna(x[generation_name]) else None, 
                axis=1
            )
            
            # Update the main dataframe with the processed slice
            self.benchmark.iloc[i:end_idx] = current_slice
            
            # Save progress if path is provided
            if save_path:
                self.benchmark.to_csv(save_path, index=False)

        return self.benchmark

    def routed_pre_generate(self,
                          weights_json_path,
                          generation_name='LLM',
                          default_value="No response available"):
        """
        Select pre-generated responses based on perspective weights from a weights configuration.
        For each prompt, selects a single perspective based on weights as probabilities
        and retrieves the response from that perspective's column.
        If the selected perspective's column doesn't exist, uses a default value.
        
        Args:
            weights_json_path (str): Path to the weights configuration JSON file
            generation_name (str): Name for the output column
            default_value (str): Default value to use if selected perspective's column doesn't exist
            
        Returns:
            pd.DataFrame: Updated benchmark with selected responses
        """
        # Load the weights configuration
        with open(weights_json_path, 'r') as f:
            weights_config = json.load(f)
        
        # Create a temporary column for tracking progress
        self.benchmark[generation_name] = None
        
        # Process each row
        for idx, row in self.benchmark.iterrows():
            concept = row['concept']
            if not concept:
                self.benchmark.at[idx, generation_name] = default_value
                continue
            
            # Get weights for this concept across all perspectives
            perspective_weights = {}
            total_weight = 0
            for perspective, config in weights_config['perspectives'].items():
                weight = config['weights'].get(concept, 0)
                if weight > 0 and perspective in self.benchmark.columns:
                    perspective_weights[perspective] = weight
                    total_weight += weight
            
            if not perspective_weights:
                self.benchmark.at[idx, generation_name] = default_value
                continue
            
            # Select a perspective based on probabilities
            import random
            selected_perspective = random.choices(
                list(perspective_weights.keys()),
                weights=list(perspective_weights.values()),
                k=1
            )[0]
            
            # Retrieve the response from the selected perspective's column
            # If the column doesn't exist or value is NaN, use default value
            if selected_perspective in self.benchmark.columns and pd.notna(row[selected_perspective]):
                self.benchmark.at[idx, generation_name] = row[selected_perspective]
            else:
                self.benchmark.at[idx, generation_name] = default_value
            
        return self.benchmark

    def ensembled_pre_generate(self,
                              weights_json_path,
                              num_generations=3,
                              ensemble_system_prompt="You are an expert at synthesizing multiple perspectives into a coherent response. Given multiple responses to the same prompt, create a balanced and comprehensive answer that captures the key points from all perspectives.",
                              generation_name='LLM',
                              default_perspective='no pre-generation'):
        """
        Select and combine pre-generated responses using an ensemble approach based on perspective weights.
        For each prompt:
        1. Selects n responses from different perspective columns based on weights
        2. Combines these responses into a template
        3. Uses a final LLM call to synthesize these responses into a coherent answer
        
        Args:
            weights_json_path (str): Path to the weights configuration JSON file
            num_generations (int): Number of responses to select and combine
            ensemble_system_prompt (str): System prompt for the final synthesis
            generation_name (str): Name for the output column
            default_perspective (str): Default perspective if no valid perspectives are found
            
        Returns:
            pd.DataFrame: Updated benchmark with combined responses
        """
        # Load the weights configuration
        with open(weights_json_path, 'r') as f:
            weights_config = json.load(f)
        
        # Create a temporary column for tracking progress
        self.benchmark[generation_name] = None
        
        # Process each row
        for idx, row in self.benchmark.iterrows():
            concept = row['concept']
            if not concept:
                self.benchmark.at[idx, generation_name] = row[default_perspective]
                continue
            
            # Get weights for this concept across all perspectives
            perspective_weights = {}
            total_weight = 0
            for perspective, config in weights_config['perspectives'].items():
                weight = config['weights'].get(concept, 0)
                if weight > 0 and perspective in self.benchmark.columns:
                    perspective_weights[perspective] = weight
                    total_weight += weight
            
            if not perspective_weights:
                self.benchmark.at[idx, generation_name] = row[default_perspective]
                continue
            
            # Select n responses using different perspective selections
            import random
            selected_responses = []
            for _ in range(num_generations):
                # Select a perspective based on probabilities
                selected_perspective = random.choices(
                    list(perspective_weights.keys()),
                    weights=list(perspective_weights.values()),
                    k=1
                )[0]
                
                # Retrieve the response from the selected perspective's column
                if pd.notna(row[selected_perspective]):
                    selected_responses.append(row[selected_perspective])
            
            if not selected_responses:
                self.benchmark.at[idx, generation_name] = row[default_perspective]
                continue
            
            # Create a template combining all responses
            ensemble_prompt = f"""Given the following {len(selected_responses)} different responses to the same prompt, create a middle answer from all perspectives:

Responses:
{chr(10).join([f'Response {i+1}: {resp}' for i, resp in enumerate(selected_responses)])}

Please synthesize these responses into a single coherent answer:"""
            
            # Generate final response using the ensemble system prompt
            from xnation.create_benchmark import create_generation_function
            generation_function = create_generation_function(
                model_name="qwen-turbo-latest",
                system_prompt=ensemble_system_prompt
            )
            final_response = generation_function(ensemble_prompt)
            
            self.benchmark.at[idx, generation_name] = final_response
            
        return self.benchmark

if __name__ == "__main__":
    # Load the benchmark data
    benchmark_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\extractions.csv"
    benchmark = pd.read_csv(benchmark_path)
    
    # Create the generator instance
    generator = ResponseGenerator(benchmark)

    # Create directory for pseudo generations if it doesn't exist
    pseudo_gen_dir = "data/xntion/pseudo_generations"
    os.makedirs(pseudo_gen_dir, exist_ok=True)
    
    # Path to the weights configuration
    weights_path = r"C:\Users\apple\PyCharmMiscProject\SAGED-Bias\data\xntion\mitigator\run_20250515_143648\optimized_weights_wasserstein_weighted_a0_b1_20250515_143649.json"
    
    # Generate responses using the routed pre-generate approach
    result = generator.routed_pre_generate(
        weights_json_path=weights_path,
        generation_name="pseudo_routed_responses"
    )
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the results
    output_path = os.path.join(pseudo_gen_dir, f"pseudo_routed_responses_{timestamp}.csv")
    result.to_csv(output_path, index=False)
    
    print(f"Pseudo generation complete. Results saved to {output_path}")

