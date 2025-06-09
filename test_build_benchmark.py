from saged._pipeline import Pipeline
from saged._database import DatabaseManager
import os
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Get absolute path for database
current_dir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(current_dir, 'data', 'db', 'saged_app.db')
db_connection = f'sqlite:///{db_path}'

# Create necessary directories
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Initialize database
db_manager = DatabaseManager(db_connection)
db_manager.initialize_database()

# Verify database connection
try:
    with db_manager.engine.connect() as conn:
        # Test connection with a simple query
        conn.execute(text("SELECT 1"))
        print("Database connection verified successfully")
except SQLAlchemyError as e:
    print(f"Database connection failed: {e}")
    raise Exception("Database connection failed. Please check your database configuration.")

# Your configuration
config = {
    'domain': 'nation',
    'concepts': ['China'],
    'branching': False,
    'branching_config': {
        'branching_pairs': '',
        'direction': 'forward',
        'source_restriction': None,
        'replacement_descriptor_require': False,
        'descriptor_threshold': 'Auto',
        'descriptor_embedding_model': 'paraphrase-Mpnet-base-v2',
        'descriptor_distance': 'cosine',
        'replacement_description': {},
        'replacement_description_saving': True,
        'replacement_description_saving_location': 'default',
        'counterfactual_baseline': True,
        'generation_function': None
    },
    'shared_config': {
        'keyword_finder': {
            'require': False,
            'reading_location': 'default',
            'method': 'embedding_on_wiki',
            'keyword_number': 7,
            'hyperlinks_info': [],
            'llm_info': {
                'n_run': 20,
                'n_keywords': 20
            },
            'max_adjustment': 150,
            'embedding_model': 'paraphrase-Mpnet-base-v2',
            'saving': True,
            'saving_location': 'nation_keywords_88c97b4a',
            'manual_keywords': None,
            'concept_keywords': None
        },
        'source_finder': {
            'require': True,
            'reading_location': 'default',
            'method': 'wiki',
            'local_file': None,
            'scrape_number': 5,
            'saving': True,
            'saving_location': 'nation_source_finder_1b6c8964',
            'scrape_backlinks': 0,
            'manual_sources': []
        },
        'scraper': {
            'require': True,
            'reading_location': 'default',
            'saving': True,
            'method': 'wiki',
            'saving_location': 'nation_scraped_sentences_3f35ac2e'
        },
        'prompt_assembler': {
            'require': True,
            'method': 'split_sentences',
            'generation_function': None,
            'keyword_list': None,
            'answer_check': False,
            'saving_location': 'nation_benchmark_4646a880',
            'max_benchmark_length': 500,
            'branching': {
                'require': False,
                'method': 'tree',
                'branching_pairs': '',
                'direction': 'forward',
                'replacement_descriptor_require': False,
                'descriptor_threshold': '0.5',
                'descriptor_embedding_model': 'paraphrase-Mpnet-base-v2',
                'descriptor_distance': 'cosine',
                'replacement_description': {},
                'replacement_description_saving': True,
                'replacement_description_saving_location': 'default',
                'counterfactual_baseline': False
            }
        }
    },
    'concept_specified_config': {
        'China': {
            'keyword_finder': {
                'manual_keywords': ['China']
            }
        }
    },
    'saving': True,
    'saving_location': 'nation_benchmark_281ac430',
    'database_config': {
        'use_database': True,
        'database_type': 'sql',
        'database_connection': db_connection,
        'table_prefix': ''
    },
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