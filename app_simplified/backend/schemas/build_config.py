from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime

class DatabaseConfig(BaseModel):
    use_database: bool = False
    database_type: str = "json"
    database_connection: str = "data/customized/database"
    table_prefix: str = ""
    source_text_table: str = "source_texts"

class FileServiceConfig(BaseModel):
    """Configuration for the file service"""
    database_config: DatabaseConfig = DatabaseConfig()
    current_table: str = "source_texts"  # Current active table name

    def update_table(self, new_table: str) -> None:
        """Update the current table name and database config"""
        self.current_table = new_table
        self.database_config.source_text_table = new_table

class LLMInquiriesConfig(BaseModel):
    n_run: int = 20
    n_keywords: int = 20
    generation_function: Optional[Any] = None
    # model_name: Optional[str] = None
    embedding_model: Optional[str] = None
    show_progress: bool = True

class KeywordFinderConfig(BaseModel):
    require: bool = True
    reading_location: str = "default"
    method: str = "embedding_on_wiki"
    keyword_number: int = 7
    hyperlinks_info: List[Any] = []
    llm_info: LLMInquiriesConfig = LLMInquiriesConfig()
    max_adjustment: int = 150
    embedding_model: str = "paraphrase-Mpnet-base-v2"
    saving: bool = True
    saving_location: str = "default"
    manual_keywords: Optional[List[str]] = None
    concept_keywords: Optional[Dict[str, List[Dict[str, str]]]] = None

class SourceFinderConfig(BaseModel):
    require: bool = True
    reading_location: str = "default"
    method: str = "wiki"
    local_file: Optional[str] = None
    scrape_number: int = 5
    saving: bool = True
    saving_location: str = "default"
    scrape_backlinks: int = 0
    manual_sources: Optional[List[str]] = None

class ScraperConfig(BaseModel):
    require: bool = True
    reading_location: str = "default"
    saving: bool = True
    method: str = "wiki"
    saving_location: str = "default"

class PromptAssemblerConfig(BaseModel):
    require: bool = True
    method: str = "split_sentences"
    generation_function: Optional[Any] = None
    keyword_list: Optional[List[str]] = None
    answer_check: bool = False
    saving_location: str = "default"
    max_benchmark_length: int = 500
    branching: Optional[Dict[str, Any]] = None

class ConceptBenchmarkConfig(BaseModel):
    keyword_finder: KeywordFinderConfig = KeywordFinderConfig()
    source_finder: SourceFinderConfig = SourceFinderConfig()
    scraper: ScraperConfig = ScraperConfig()
    prompt_assembler: PromptAssemblerConfig = PromptAssemblerConfig()

class BranchingConfig(BaseModel):
    branching_pairs: str = "not_all"
    direction: str = "both"
    source_restriction: Optional[str] = None
    replacement_descriptor_require: bool = False
    descriptor_threshold: str = "Auto"
    descriptor_embedding_model: str = "paraphrase-Mpnet-base-v2"
    descriptor_distance: str = "cosine"
    replacement_description: Dict[str, Any] = {}
    replacement_description_saving: bool = True
    replacement_description_saving_location: str = "data/customized/benchmark/replacement_description.json"
    counterfactual_baseline: bool = True
    generation_function: Optional[Any] = None

class DomainBenchmarkConfig(BaseModel):
    domain: str
    concepts: List[str]
    branching: bool = False
    branching_config: Optional[BranchingConfig] = None
    shared_config: ConceptBenchmarkConfig = ConceptBenchmarkConfig()
    concept_specified_config: Dict[str, Any] = {}
    saving: bool = True
    saving_location: str = "default"
    database_config: DatabaseConfig = DatabaseConfig()

class AnalyticsConfig(BaseModel):
    database_config: DatabaseConfig = DatabaseConfig()
    generation: Dict[str, Any] = {
        "require": True,
        "generate_dict": {},
        "generation_saving_location": 'data/customized/_sbg_benchmark.csv',
        "generation_list": [],
        "baseline": 'baseline',
    }
    extraction: Dict[str, Any] = {
        "feature_extractors": [
            'personality_classification',
            'toxicity_classification',
            'sentiment_classification',
            'stereotype_classification',
            'regard_classification'
        ],
        'extractor_configs': {},
        "calibration": True,
        "extraction_saving_location": 'data/customized/_sbge_benchmark.csv',
    }
    analysis: Dict[str, Any] = {
        "specifications": ['concept', 'source_tag'],
        "analyzers": ['mean', 'selection_rate', 'precision'],
        "analyzer_configs": {
            'selection_rate': {'standard_by': 'mean'},
            'precision': {'tolerance': 0.1}
        },
        'statistics_saving_location': 'data/customized/_sbgea_statistics.csv',
        "disparity_saving_location": 'data/customized/_sbgea_disparity.csv',
    }

# Database Models
class KeywordsData(BaseModel):
    id: Optional[int]
    domain: str
    concept: str
    keywords: Dict[str, Dict[str, Any]]
    created_at: datetime

class SourceFinderData(BaseModel):
    id: Optional[int]
    domain: str
    concept: str
    concept_shared_source: List[Dict[str, Any]]
    keywords: Dict[str, Dict[str, Any]]
    created_at: datetime

class ScrapedSentencesData(BaseModel):
    id: Optional[int]
    domain: str
    concept: str
    concept_shared_source: List[Dict[str, Any]]
    keywords: Dict[str, Dict[str, Any]]
    created_at: datetime

class SplitSentencesData(BaseModel):
    id: Optional[int]
    domain: str
    concept: str
    keyword: str
    prompts: str
    baseline: str
    source_tag: str
    created_at: datetime

class QuestionsData(BaseModel):
    id: Optional[int]
    domain: str
    concept: str
    keyword: str
    prompts: str
    baseline: str
    source_tag: str
    created_at: datetime

class ReplacementDescriptionData(BaseModel):
    id: Optional[int]
    domain: str
    concept: str
    replacement_description: Dict[str, Any]
    created_at: datetime

class BenchmarkData(BaseModel):
    id: Optional[int]
    domain: str
    concept: str
    keyword: Optional[str]
    prompts: Optional[str]
    baseline: Optional[str]
    source_tag: Optional[str]
    data: Optional[Dict[str, Any]]
    created_at: datetime

class AllDataTiersResponse(BaseModel):
    keywords: Optional[KeywordsData]
    source_finder: Optional[SourceFinderData]
    scraped_sentences: Optional[ScrapedSentencesData]
    split_sentences: Optional[SplitSentencesData]
    questions: Optional[QuestionsData]
    replacement_description: Optional[ReplacementDescriptionData]
    benchmark: Optional[BenchmarkData]

class BenchmarkMetadata(BaseModel):
    """Schema for benchmark metadata stored in the database"""
    id: Optional[int]
    domain: str
    data: Optional[Dict[str, Any]]
    table_names: Dict[str, Optional[str]]
    configuration: Dict[str, Any]
    database_config: Dict[str, Any]
    time_stamp: str
    created_at: Optional[datetime]

class BenchmarkResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    database_data: Optional[AllDataTiersResponse] = None 