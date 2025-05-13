from ._utility import check_benchmark, check_generation_function
import pandas as pd
import os
from tqdm import tqdm

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

