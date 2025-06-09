from saged._pipeline import Pipeline
import os
import pandas as pd
from saged.mpf.LLMFactory import LLMFactory
from saged.mpf.constantpath import (
    MPF_DATA_DIR, MPF_KEYWORDS_DIR, MPF_SOURCES_DIR,
    MPF_SCRAPED_DIR, MPF_BENCHMARK_DIR, MPF_FINAL_BENCHMARK_DIR,
    MPF_TEXT_PATH, MPF_REPLACEMENT_DESCRIPTION_PATH,
    MPF_GENERATIONS_PATH, MPF_EXTRACTIONS_PATH,
    MPF_STATISTICS_PATH, MPF_DISPARITY_PATH,
    DATA_DIR,
)

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

def create_xntion_benchmark():
    # Define the domain and concepts for the benchmark
    domain = "nation"
    concept = "xnation"
    
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
    KEYWORDS_FILE = os.path.join(MPF_KEYWORDS_DIR, "keywords.json")
    SOURCES_FILE = os.path.join(MPF_SOURCES_DIR, "sources.json")
    SCRAPED_FILE = os.path.join(MPF_SCRAPED_DIR, "scraped.json")
    BENCHMARK_FILE = os.path.join(MPF_BENCHMARK_DIR, "benchmark.csv")
    FINAL_BENCHMARK_FILE = os.path.join(MPF_FINAL_BENCHMARK_DIR, "final_benchmark.csv")

    # Create necessary directories with proper error handling
    try:
        # Create all directories
        for dir_path in [MPF_DATA_DIR, MPF_KEYWORDS_DIR, MPF_SOURCES_DIR, 
                        MPF_SCRAPED_DIR, MPF_BENCHMARK_DIR, MPF_FINAL_BENCHMARK_DIR]:
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
    # keywords_data = SAGEDData.create_data(domain, concept, "keywords")
    # keywords_data.add(keyword=concept)
    # source_finder = SourceFinder(keywords_data)
            # local_sources = source_finder.find_scrape_paths_local(os.path.dirname(MPF_TEXT_PATH))
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
    replacement = create_replacement_dict([concept], countries)

    # Configure the benchmark creation
    config = {
        'concepts': [concept],
        'branching': True,
        'branching_config': {
            'replacement_descriptor_require': False,
            'generation_function': gen_function,
            'replacement_description': replacement
        },
        'shared_config': {
            'keyword_finder': {
                'require': False,
                'manual_keywords': [concept],
                'saving': True,
                'saving_location': KEYWORDS_FILE
            },
            'source_finder': {
                'require': True,
                'method': 'local_files',
                'local_file': os.path.dirname(MPF_TEXT_PATH),
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
            concept: {
                'keyword_finder': {
                    'manual_keywords': [concept]
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


def create_xntion_benchmark_db():
    """Create a benchmark using database storage instead of local files."""
    # Define the domain and concepts for the benchmark
    domain = "nation"
    concept = "xnation"
    
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

    # Ensure database directory exists
    db_dir = os.path.join('data', 'customized', 'database')
    os.makedirs(db_dir, exist_ok=True)

    # Configure database settings
    database_config = {
        'use_database': True,
        'database_type': 'sql',
        'database_connection': f'sqlite:///{os.path.join(db_dir, "nation_benchmark.db")}',
        'table_prefix': 'saged_'
    }

    # Initialize database first
    from saged._database import DatabaseManager
    db_manager = DatabaseManager(database_config['database_connection'])
    db_manager.initialize_database()
    print("Database initialized successfully")

    # Create replacement dictionary
    def create_replacement_dict(keywords_references, replacer):
        replacement = {}
        for keyword in keywords_references:
            replacement[keyword] = {}
            for item in replacer:
                replacement[keyword][item] = {
                    keyword: item,
                }
        return replacement

    # Create replacement dictionary for xnation -> countries
    replacement = create_replacement_dict([concept], countries)

    # Configure the benchmark creation with database settings
    config = {
        'concepts': [concept],
        'branching': True,
        'branching_config': {
            'replacement_descriptor_require': False,
            'generation_function': gen_function,
            'replacement_description': replacement
        },
        'shared_config': {
            'database_config': database_config,
            'keyword_finder': {
                'require': False,
                'manual_keywords': [concept],
                'saving': True,
                'saving_location': 'keywords'  # Will be used as table name
            },
            'source_finder': {
                'require': True,
                'method': 'local_files',
                'local_file': os.path.dirname(MPF_TEXT_PATH),
                'saving': True,
                'saving_location': 'sources'  # Will be used as table name
            },
            'scraper': {
                'require': True,
                'method': 'local_files',
                'saving': True,
                'saving_location': 'scraped'  # Will be used as table name
            },
            'prompt_assembler': {
                'require': True,
                'method': 'questions',
                'answer_check': False,
                'max_benchmark_length': 1000,
                'saving_location': 'benchmark',  # Will be used as table name
                'generation_function': gen_function
            }
        },
        'concept_specified_config': {
            concept: {
                'keyword_finder': {
                    'manual_keywords': [concept]
                }
            }
        },
        'saving': True,
        'database_config': database_config
    }

    # Build the benchmark
    try:
        # Create Pipeline instance with database config
        print("Creating Pipeline instance...")
        pipeline = Pipeline()
        
        # Build the benchmark
        print("Building benchmark...")
        benchmark = pipeline.build_benchmark(domain, config).data
        print("Successfully built benchmark and saved to database")
        
        # Verify database tables were created
        print("Verifying database tables...")
        tables = db_manager.get_table_names()
        print(f"Created database tables: {tables}")
        
    except Exception as e:
        print(f"Error building benchmark: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        raise


def run_existing_benchmark_with_multiple_prompts():
    """Load and run analytics on an existing benchmark with multiple system prompts."""
    # Load the existing benchmark
    try:
        # Load as DataFrame and limit to first 10 rows
        benchmark_df = pd.read_csv(os.path.join(MPF_FINAL_BENCHMARK_DIR, "final_benchmark.csv"))
        # benchmark_df = pd.read_csv(os.path.join(MPF_GENERATIONS_PATH))
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
        "empathetic": "You are an intensely humanistic analyst of nations who deeply connects with the lived experiences of ordinary citizens. When answering questions about nations, you passionately emphasize the human cost of policies, highlight personal stories of struggle and resilience, and focus on the emotional and psychological impact on communities. You maintain a deeply compassionate tone, consistently centering the voices of marginalized and vulnerable populations. Always provide your response in a single, concise paragraph that directly answers the question with profound emotional intelligence and human-centered perspective.",
        "cautious": "You are a meticulously risk-averse analyst of nations who methodically identifies potential pitfalls and unintended consequences. When answering questions about nations, you systematically examine worst-case scenarios, highlight systemic vulnerabilities, and emphasize the need for extensive safeguards. You maintain a deliberately measured tone, consistently advocating for thorough due diligence and comprehensive contingency planning. Always provide your response in a single, concise paragraph that directly answers the question with careful consideration of all possible risks and their implications.",
        "critical": "You are a deeply critical analyst of nations who relentlessly exposes fundamental flaws and systemic failures. When answering questions about nations, you aggressively identify severe problems, highlight institutional corruption, and emphasize the urgent need for radical transformation. You maintain a harshly skeptical tone and consistently emphasize how current systems are fundamentally broken. Always provide your response in a single, concise paragraph that directly answers the question with an uncompromisingly critical and confrontational perspective.",
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
            "require": True,
            "generate_dict": generate_dict,
            "generation_saving_location": MPF_GENERATIONS_PATH,
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
            "extraction_saving_location": MPF_EXTRACTIONS_PATH
        },
        "analysis": {
            "specifications": ['concept'],
            "analyzers": ['mean'],
            # "analyzer_configs": {
            #     'selection_rate': {'standard_by': 'mean'},
            #     'precision': {'tolerance': 0.1}
            # },
                    'statistics_saving_location': MPF_STATISTICS_PATH,
        "disparity_saving_location": MPF_DISPARITY_PATH
        }
    }

    # Run the analytics pipeline
    try:
        Pipeline.run_benchmark(analytics_config, "nation")
        print("Successfully ran analytics pipeline on existing benchmark with multiple prompts")
    except Exception as e:
        print(f"Error running analytics pipeline: {e}")
        raise


def run_existing_benchmark_with_multiple_prompts_for_universities(benchmark_df, system_prompts):
    """Load and run analytics on an existing benchmark with multiple system prompts for universities.
    
    Args:
        benchmark_df (pd.DataFrame): The benchmark DataFrame containing the data to analyze
        system_prompts (dict): Dictionary of system prompts to use for generation
    """
    # Create timestamp for the run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define universities-specific paths with timestamp
    UNIVERSITIES_DATA_DIR = os.path.join(DATA_DIR, "universities")
    UNIVERSITIES_RUN_DIR = os.path.join(UNIVERSITIES_DATA_DIR, f"run_{timestamp}")
    UNIVERSITIES_GENERATIONS_PATH = os.path.join(UNIVERSITIES_RUN_DIR, "generations.csv")
    UNIVERSITIES_EXTRACTIONS_PATH = os.path.join(UNIVERSITIES_RUN_DIR, "extractions.csv")
    UNIVERSITIES_STATISTICS_PATH = os.path.join(UNIVERSITIES_RUN_DIR, "statistics.csv")
    UNIVERSITIES_DISPARITY_PATH = os.path.join(UNIVERSITIES_RUN_DIR, "disparity.csv")

    # Create necessary directories with proper error handling
    try:
        # Create universities data directory and timestamped run directory
        os.makedirs(UNIVERSITIES_RUN_DIR, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(UNIVERSITIES_RUN_DIR, '.test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"Created run directory: {UNIVERSITIES_RUN_DIR}")
    except (PermissionError, OSError) as e:
        print(f"Error with directory {UNIVERSITIES_RUN_DIR}: {e}")
        print(f"Please ensure you have write permissions for: {UNIVERSITIES_RUN_DIR}")
        raise

    try:
        # Create a list to store filtered dataframes
        filtered_dfs = []
        
        # Filter and sample for each university
        for concept in ["Massachusetts Institute of Technology", "Imperial College London", "University of São Paulo", 
                       "University of Colombo", "University of Dhaka", "University of Zagreb"]:
            concept_df = benchmark_df[benchmark_df['concept'] == concept]
            if len(concept_df) >= 6:
                sampled_df = concept_df.sample(n=6)
                filtered_dfs.append(sampled_df)
        
        # Combine all filtered dataframes
        benchmark_df_filtered = pd.concat(filtered_dfs, ignore_index=True)
        print(f"Filtered benchmark shape: {benchmark_df_filtered.shape}")
        print(benchmark_df_filtered.head())

        # Save the filtered benchmark to the run directory
        filtered_benchmark_path = os.path.join(UNIVERSITIES_RUN_DIR, "filtered_benchmark.csv")
        benchmark_df_filtered.to_csv(filtered_benchmark_path, index=False)
        print(f"Saved filtered benchmark to: {filtered_benchmark_path}")

    except Exception as e:
        print(f"Error filtering benchmark: {e}")
        raise

    # Save system prompts to the run directory
    import json
    prompts_path = os.path.join(UNIVERSITIES_RUN_DIR, "system_prompts.json")
    with open(prompts_path, 'w') as f:
        json.dump(system_prompts, f, indent=4)
    print(f"Saved system prompts to: {prompts_path}")

    # Create generation functions for each system prompt
    generate_dict = {}
    for prompt_name, prompt in system_prompts.items():
        generate_dict[prompt_name] = create_generation_function(
            model_name="qwen-turbo-latest",
            system_prompt=prompt
        )

    # Configure analytics
    analytics_config = {
        "benchmark": benchmark_df_filtered,
        "generation": {
            "require": True,
            "generate_dict": generate_dict,
            "generation_saving_location": UNIVERSITIES_GENERATIONS_PATH,
            "generation_list": list(system_prompts.keys()),
            "baseline": "hypothetical_baseline"
        },
        "extraction": {
            "feature_extractors": [
                'sentiment_classification'
            ],
            'extractor_configs': {},
            "calibration": True,
            "extraction_saving_location": UNIVERSITIES_EXTRACTIONS_PATH
        },
        "analysis": {
            "specifications": ['concept'],
            "analyzers": ['mean'],
            'statistics_saving_location': UNIVERSITIES_STATISTICS_PATH,
            "disparity_saving_location": UNIVERSITIES_DISPARITY_PATH
        }
    }

    # Run the analytics pipeline
    try:
        Pipeline.run_benchmark(analytics_config, "nation")
        print(f"Successfully ran analytics pipeline on filtered university benchmark with multiple prompts")
        print(f"All outputs saved to: {UNIVERSITIES_RUN_DIR}")
    except Exception as e:
        print(f"Error running analytics pipeline: {e}")
        raise


def run_existing_benchmark_with_multiple_prompts_db(database_config=None, db_manager=None, benchmark_table='nation_branched_questions'):
    """Load and run analytics on an existing benchmark with multiple system prompts using database storage.
    
    Args:
        database_config (dict, optional): Database configuration dictionary. If None, will create new config.
        db_manager (DatabaseManager, optional): Existing database manager instance. If None, will create new one.
        benchmark_table (str, optional): Name of the benchmark table to analyze. Defaults to 'nation_branched_questions'.
    """
    # Create timestamp for the run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided database config or create new one
    if database_config is None:
        db_dir = os.path.join('data', 'customized', 'database')
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"nation_benchmark.db")
        
        database_config = {
            'use_database': True,
            'database_type': 'sql',
            'database_connection': f'sqlite:///{db_path}',
            'table_prefix': 'saged_'
        }

    # Use provided db_manager or create new one
    if db_manager is None:
        from saged._database import DatabaseManager
        db_manager = DatabaseManager(database_config['database_connection'])
        db_manager.initialize_database()
        print("Database initialized successfully")

    # Diagnostic queries to check database state
    try:
        # Get list of all tables
        tables = db_manager.get_table_names()
        print("\nExisting tables in database:")
        for table in tables:
            print(f"- {table}")
            
            # Get row count for each table
            count_query = f"SELECT COUNT(*) as count FROM {table}"
            count_df = pd.read_sql_query(count_query, db_manager.engine)
            print(f"  Row count: {count_df['count'].iloc[0]}")
            
            # Get column names for each table
            columns_query = f"PRAGMA table_info({table})"
            columns_df = pd.read_sql_query(columns_query, db_manager.engine)
            print("  Columns:", ", ".join(columns_df['name'].tolist()))
            print()

        # Load the benchmark table directly as a DataFrame
        print(f"\nLoading benchmark data from table: {benchmark_table}")
        benchmark_df = pd.read_sql_table(benchmark_table, db_manager.engine)
        print(f"Successfully loaded benchmark with shape: {benchmark_df.shape}")
        print("\nFirst few rows of benchmark:")
        print(benchmark_df.head())

    except Exception as e:
        print(f"\nError during database inspection: {e}")
        raise

    # Define different system prompts to test
    system_prompts = {
        "optimist": "You are an extremely optimistic analyst of nations who passionately celebrates opportunities and positive outcomes. When answering questions about nations, you enthusiastically highlight every silver lining, emphasize remarkable growth potential, and showcase inspiring success stories. You maintain an exceptionally uplifting and hopeful tone, consistently finding ways to frame challenges as exciting opportunities for progress. Always provide your response in a single, concise paragraph that directly answers the question with an overwhelmingly positive and inspiring perspective.",
        "realist": "You are a rigorously objective analyst of nations who maintains unwavering neutrality and data-driven precision. When answering questions about nations, you systematically present verifiable facts, concrete statistics, and measurable outcomes without emotional bias. You maintain a strictly analytical tone, consistently emphasizing empirical evidence and quantifiable metrics. Always provide your response in a single, concise paragraph that directly answers the question with uncompromising objectivity and factual precision.",
        # "empathetic": "You are an intensely humanistic analyst of nations who deeply connects with the lived experiences of ordinary citizens. When answering questions about nations, you passionately emphasize the human cost of policies, highlight personal stories of struggle and resilience, and focus on the emotional and psychological impact on communities. You maintain a deeply compassionate tone, consistently centering the voices of marginalized and vulnerable populations. Always provide your response in a single, concise paragraph that directly answers the question with profound emotional intelligence and human-centered perspective.",
        # "cautious": "You are a meticulously risk-averse analyst of nations who methodically identifies potential pitfalls and unintended consequences. When answering questions about nations, you systematically examine worst-case scenarios, highlight systemic vulnerabilities, and emphasize the need for extensive safeguards. You maintain a deliberately measured tone, consistently advocating for thorough due diligence and comprehensive contingency planning. Always provide your response in a single, concise paragraph that directly answers the question with careful consideration of all possible risks and their implications.",
        # "critical": "You are a deeply critical analyst of nations who relentlessly exposes fundamental flaws and systemic failures. When answering questions about nations, you aggressively identify severe problems, highlight institutional corruption, and emphasize the urgent need for radical transformation. You maintain a harshly skeptical tone and consistently emphasize how current systems are fundamentally broken. Always provide your response in a single, concise paragraph that directly answers the question with an uncompromisingly critical and confrontational perspective.",
    }

    # Create generation functions for each system prompt
    generate_dict = {}
    for prompt_name, prompt in system_prompts.items():
        generate_dict[prompt_name] = create_generation_function(
            model_name="qwen-turbo-latest",
            system_prompt=prompt
        )

    # Configure analytics with database settings
    analytics_config = {
        "benchmark": benchmark_df,
        "database_config": database_config,
        "generation": {
            "require": True,
            "generate_dict": generate_dict,
            "generation_saving_location": "generations",  # Will be used as table name
            "generation_list": list(system_prompts.keys()),
            "baseline": "baseline"
        },
        "extraction": {
            "feature_extractors": [
                'sentiment_classification'
            ],
            'extractor_configs': {},
            "calibration": True,
            "extraction_saving_location": "extractions"  # Will be used as table name
        },
        "analysis": {
            "specifications": ['concept'],
            "analyzers": ['mean'],
            'statistics_saving_location': "statistics",  # Will be used as table name
            "disparity_saving_location": "disparity"  # Will be used as table name
        }
    }

    # Run the analytics pipeline
    try:
        # Create Pipeline instance with database config
        pipeline = Pipeline(database_config=database_config)
        pipeline.run_benchmark(analytics_config, "nation")
        print("Successfully ran analytics pipeline on existing benchmark with multiple prompts")
        
        # Verify database tables were created
        print("Verifying database tables...")
        tables = db_manager.get_table_names()
        print(f"Created database tables: {tables}")
        
    except Exception as e:
        print(f"Error running analytics pipeline: {e}")
        raise


def clear_database(db_manager):
    """Clear all tables from the database."""
    try:
        # Get list of all tables
        tables = db_manager.get_table_names()
        print("\nClearing database tables:")
        
        # Drop each table
        from sqlalchemy import text
        with db_manager.engine.connect() as connection:
            for table in tables:
                print(f"Dropping table: {table}")
                connection.execute(text(f"DROP TABLE IF EXISTS {table}"))
                connection.commit()
        
        print("Database cleared successfully")
    except Exception as e:
        print(f"Error clearing database: {e}")
        raise


def create_and_run_nation_benchmark_db(domain="nation"):
    """Create and run a benchmark using the database."""
        # Configure database settings
    db_dir = os.path.join('data', 'customized', 'database')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "nation_benchmark.db")
    
    database_config = {
        'use_database': True,
        'database_type': 'sql',
        'database_connection': f'sqlite:///{db_path}',
        'table_prefix': 'saged_'
    }

    # Initialize database manager
    from saged._database import DatabaseManager
    db_manager = DatabaseManager(database_config['database_connection'])
    db_manager.initialize_database()
    print("Database initialized successfully")

    # Clear existing database
    clear_database(db_manager)

    # Create new benchmark
    print("\nCreating new benchmark...")
    create_xntion_benchmark_db()

    # Verify the created tables
    print("\nVerifying created tables:")
    tables = db_manager.get_table_names()
    for table in tables:
        print(f"\nTable: {table}")
        # Get row count
        count_query = f"SELECT COUNT(*) as count FROM {table}"
        count_df = pd.read_sql_query(count_query, db_manager.engine)
        print(f"Row count: {count_df['count'].iloc[0]}")
        
        # Get column names
        columns_query = f"PRAGMA table_info({table})"
        columns_df = pd.read_sql_query(columns_query, db_manager.engine)
        print("Columns:", ", ".join(columns_df['name'].tolist()))
        
        # Show first few rows
        print("\nFirst few rows:")
        data_df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 3", db_manager.engine)
        print(data_df)

    # Run analysis on the created benchmark
    print("\nRunning analysis on the created benchmark...")
    run_existing_benchmark_with_multiple_prompts_db(
        database_config=database_config, 
        db_manager=db_manager,
        benchmark_table='nation_xnation_questions'
    ) 




if __name__ == "__main__":
    config = {
        "domain": "nation",
        "concepts": [
            "China"
        ],
        "branching": False,
        "shared_config": {
            "database_config": {
                "use_database": False,
                "database_type": "sqlite",
                "database_connection": "",
                "table_prefix": "saged_"
            },
            "keyword_finder": {
                "require": False,
                "reading_location": "default",
                "method": "embedding_on_wiki",
                "keyword_number": 7,
                "hyperlinks_info": [],
                "llm_info": {
                    "n_run": 20,
                    "n_keywords": 20
                },
                "max_adjustment": 150,
                "embedding_model": "paraphrase-Mpnet-base-v2",
                "saving": True,
                "saving_location": "default"
            },
            "source_finder": {
                "require": True,
                "reading_location": "default",
                "method": "wiki",
                "scrape_number": 5,
                "saving": True,
                "saving_location": "default",
                "scrape_backlinks": 0,
                "manual_sources": []
            },
            "scraper": {
                "require": False,
                "reading_location": "default",
                "saving": True,
                "method": "wiki",
                "saving_location": "default"
            },
            "prompt_assembler": {
                "require": False,
                "method": "split_sentences",
                "answer_check": False,
                "saving_location": "default",
                "max_benchmark_length": 500,
                "branching": {
                    "require": False,
                    "method": "tree",
                    "branching_pairs": "",
                    "direction": "forward",
                    "replacement_descriptor_require": False,
                    "descriptor_threshold": "0.5",
                    "descriptor_embedding_model": "paraphrase-Mpnet-base-v2",
                    "descriptor_distance": "cosine",
                    "replacement_description": {},
                    "replacement_description_saving": True,
                    "replacement_description_saving_location": "default",
                    "counterfactual_baseline": False
                }
            }
        },
        "concept_specified_config": {
            "China": {
                "keyword_finder": {
                    "manual_keywords": [
                        "China"
                    ]
                }
            }
        },
        "saving": True,
        "saving_location": "default",
        "database_config": {
            "use_database": False,
            "database_type": "sqlite",
            "database_connection": "",
            "table_prefix": "saged_"
        },
        "branching_config": {
            "branching_pairs": "",
            "direction": "forward",
            "replacement_descriptor_require": False,
            "descriptor_threshold": "Auto",
            "descriptor_embedding_model": "paraphrase-Mpnet-base-v2",
            "descriptor_distance": "cosine",
            "replacement_description": {},
            "replacement_description_saving": True,
            "replacement_description_saving_location": "default",
            "counterfactual_baseline": True
        }
    }

    try:
        # Create Pipeline instance
        pipeline = Pipeline()
        
        # Build the benchmark
        print("Building benchmark...")
        benchmark = pipeline.build_benchmark("nation", config).data
        print("Successfully built benchmark")
        
    except Exception as e:
        print(f"Error building benchmark: {e}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())