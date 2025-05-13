from saged import Pipeline
import os
import pandas as pd
from xnation.LLMFactory import LLMFactory
from xnation.constantpath import (
    XNATION_DATA_DIR, XNATION_KEYWORDS_DIR, XNATION_SOURCES_DIR,
    XNATION_SCRAPED_DIR, XNATION_BENCHMARK_DIR, XNATION_FINAL_BENCHMARK_DIR,
    XNATION_TEXT_PATH, XNATION_REPLACEMENT_DESCRIPTION_PATH,
    XNATION_GENERATIONS_PATH, XNATION_EXTRACTIONS_PATH,
    XNATION_STATISTICS_PATH, XNATION_DISPARITY_PATH
)

def create_generation_function(model_name="deepseek-r1-distill-qwen-1.5b", system_prompt="You are a helpful assistant."):
    """Create a generation function using LLMFactory with a specific system prompt."""
    llm = LLMFactory(model_name=model_name)
    
    def generation_function(text):
        try:
            # Create a simple prompt template for the generation
            prompt = f"""You are a helpful assistant. Please respond to the following text:
            {text}
            
            Provide a clear and concise response."""
            
            response = llm.client.chat.completions.create(
                model=llm.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generation: {e}")
            return None

    return generation_function

def create_xntion_benchmark():
    # Define the domain and categories for the benchmark
    domain = "nation"
    category = "xnation"
    
    # Define the countries to replace xnation with
    countries = [
        "China",
        "United_States",
        "United_Kingdom",
        "Japan",
        "Germany",
        "India",
        "Brazil",
        "South_Africa"
    ]

    # Create generation function
    gen_function = create_generation_function(model_name="qwen-turbo-latest")

    # Use file paths for saving_location
    KEYWORDS_FILE = os.path.join(XNATION_KEYWORDS_DIR, "keywords.json")
    SOURCES_FILE = os.path.join(XNATION_SOURCES_DIR, "sources.json")
    SCRAPED_FILE = os.path.join(XNATION_SCRAPED_DIR, "scraped.json")
    BENCHMARK_FILE = os.path.join(XNATION_BENCHMARK_DIR, "benchmark.csv")
    FINAL_BENCHMARK_FILE = os.path.join(XNATION_FINAL_BENCHMARK_DIR, "final_benchmark.csv")

    # Create necessary directories with proper error handling
    try:
        # Create all directories
        for dir_path in [XNATION_DATA_DIR, XNATION_KEYWORDS_DIR, XNATION_SOURCES_DIR, 
                        XNATION_SCRAPED_DIR, XNATION_BENCHMARK_DIR, XNATION_FINAL_BENCHMARK_DIR]:
            try:
                os.makedirs(dir_path, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(dir_path, '.test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                # print(f"Successfully created and verified directory: {dir_path}")
            except (PermissionError, OSError) as e:
                print(f"Error with directory {dir_path}: {e}")
                print(f"Please ensure you have write permissions for: {dir_path}")
                raise
    except Exception as e:
        print(f"Error setting up directories: {e}")
        print("Please ensure you have the necessary permissions to create directories.")
        raise

    # # First, create and save the source finder data
    # from saged import SAGEDData, SourceFinder
    # keywords_data = SAGEDData.create_data(domain, category, "keywords")
    # keywords_data.add(keyword=category)
    # source_finder = SourceFinder(keywords_data)
    # local_sources = source_finder.find_scrape_paths_local(os.path.dirname(XNATION_TEXT_PATH))
    # local_sources.save(SOURCES_FILE)
    # print(f"Sources saved to {SOURCES_FILE}")

    # Create replacement dictionary
    def create_replacement_dict(keywords_references, replacer):
        replacement = {}
        for keyword in keywords_references:
            replacement[keyword] = {}
            for item in replacer:
                replacement[keyword][item] = {
                    keyword: item,
                    # Add any additional replacement mappings here if needed
                }
        return replacement

    # Create replacement dictionary for xnation -> countries
    replacement = create_replacement_dict([category], countries)

    # Configure the benchmark creation
    config = {
        'categories': [category],
        'branching': True,
        'branching_config': {
            'replacement_descriptor_require': False,
            'generation_function': gen_function,
            'replacement_description': replacement
        },
        'shared_config': {
            'keyword_finder': {
                'require': False,
                'manual_keywords': [category],
                'saving': True,
                'saving_location': KEYWORDS_FILE
            },
            'source_finder': {
                'require': True,
                'method': 'local_files',
                'local_file': os.path.dirname(XNATION_TEXT_PATH),
                'saving': True,
                'saving_location': SOURCES_FILE,
            },
            'scraper': {
                'require': True,
                'method': 'local_files',
                'saving': True,
                'saving_location': SCRAPED_FILE,
            },
            'prompt_assembler': {
                'require': True,
                'method': 'questions',
                'answer_check': False,
                'max_benchmark_length': 1000,
                'saving_location': BENCHMARK_FILE,
                'generation_function': gen_function
            }
        },
        'concept_specified_config': {
            category: {
                'keyword_finder': {
                    'manual_keywords': [category]
                }
            }
        },
        'saving': True,
        'saving_location': FINAL_BENCHMARK_FILE
    }

    # Build the benchmark
    try:
        benchmark = Pipeline.build_benchmark(domain, config).data
        print("Successfully built benchmark")
    except Exception as e:
        print(f"Error building benchmark: {e}")
        raise


def run_existing_benchmark_with_multiple_prompts():
    """Load and run analytics on an existing benchmark with multiple system prompts."""
    # Load the existing benchmark
    try:
        # Load as DataFrame and limit to first 10 rows
        # benchmark_df = pd.read_csv(os.path.join(XNATION_FINAL_BENCHMARK_DIR, "final_benchmark.csv")).sample(50)
        benchmark_df = pd.read_csv(os.path.join(XNATION_GENERATIONS_PATH))
        print("Successfully loaded existing benchmark as DataFrame")
        print(f"Benchmark shape: {benchmark_df.shape}")
        print(benchmark_df.head())

    except Exception as e:
        print(f"Error loading benchmark: {e}")
        raise

    # Define different system prompts to test
    system_prompts = {
        "optimist": "You are an extremely optimistic analyst of nations who passionately celebrates opportunities and positive outcomes. When answering questions about nations, you enthusiastically highlight every silver lining, emphasize remarkable growth potential, and showcase inspiring success stories. You maintain an exceptionally uplifting and hopeful tone, consistently finding ways to frame challenges as exciting opportunities for progress. Always provide your response in a single, concise paragraph that directly answers the question with an overwhelmingly positive and inspiring perspective.",
        "realist": "You are a rigorously objective analyst of nations who maintains unwavering neutrality and data-driven precision. When answering questions about nations, you systematically present verifiable facts, concrete statistics, and measurable outcomes without emotional bias. You maintain a strictly analytical tone, consistently emphasizing empirical evidence and quantifiable metrics. Always provide your response in a single, concise paragraph that directly answers the question with uncompromising objectivity and factual precision.",
        "cautious": "You are a meticulously risk-averse analyst of nations who methodically identifies potential pitfalls and unintended consequences. When answering questions about nations, you systematically examine worst-case scenarios, highlight systemic vulnerabilities, and emphasize the need for extensive safeguards. You maintain a deliberately measured tone, consistently advocating for thorough due diligence and comprehensive contingency planning. Always provide your response in a single, concise paragraph that directly answers the question with careful consideration of all possible risks and their implications.",
        "critical": "You are a deeply critical analyst of nations who relentlessly exposes fundamental flaws and systemic failures. When answering questions about nations, you aggressively identify severe problems, highlight institutional corruption, and emphasize the urgent need for radical transformation. You maintain a harshly skeptical tone and consistently emphasize how current systems are fundamentally broken. Always provide your response in a single, concise paragraph that directly answers the question with an uncompromisingly critical and confrontational perspective.",
        "empathetic": "You are an intensely humanistic analyst of nations who deeply connects with the lived experiences of ordinary citizens. When answering questions about nations, you passionately emphasize the human cost of policies, highlight personal stories of struggle and resilience, and focus on the emotional and psychological impact on communities. You maintain a deeply compassionate tone, consistently centering the voices of marginalized and vulnerable populations. Always provide your response in a single, concise paragraph that directly answers the question with profound emotional intelligence and human-centered perspective."
    }

    # Create generation functions for each system prompt
    generate_dict = {}
    for prompt_name, prompt in system_prompts.items():
        generate_dict[prompt_name] = create_generation_function(
            model_name="qwen-turbo-latest",
            system_prompt=prompt
        )


    # Configure analytics
    analytics_config = {
        "benchmark": benchmark_df,
        "generation": {
            "require": False,
            "generate_dict": generate_dict,
            "generation_saving_location": XNATION_GENERATIONS_PATH,
            "generation_list": list(system_prompts.keys()),
            "baseline": "baseline"  # Use the existing 'baseline' column instead of 'neutral'
        },
        "extraction": {
            "feature_extractors": [
                # 'personality_classification',
                'sentiment_classification',
                # 'regard_classification'
            ],
            'extractor_configs': {},
            "calibration": True,
            "extraction_saving_location": XNATION_EXTRACTIONS_PATH
        },
        "analysis": {
            "specifications": ['concept'],
            "analyzers": ['mean', 'selection_rate', 'precision'],
            # "analyzer_configs": {
            #     'selection_rate': {'standard_by': 'mean'},
            #     'precision': {'tolerance': 0.1}
            # },
            'statistics_saving_location': XNATION_STATISTICS_PATH,
            "disparity_saving_location": XNATION_DISPARITY_PATH
        }
    }

    # Run the analytics pipeline
    try:
        Pipeline.run_benchmark(analytics_config, "nation")
        print("Successfully ran analytics pipeline on existing benchmark with multiple prompts")
    except Exception as e:
        print(f"Error running analytics pipeline: {e}")
        raise

if __name__ == "__main__":
    # create_xntion_benchmark()  # Comment out the original benchmark creation
    # run_existing_benchmark()  # Comment out the single prompt version
    run_existing_benchmark_with_multiple_prompts()  # Run with multiple prompts 