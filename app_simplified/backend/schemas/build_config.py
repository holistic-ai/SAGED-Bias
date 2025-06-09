from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime

class DatabaseConfig(BaseModel):
    use_database: bool = True
    database_type: str = "sqlite"
    database_connection: str = "sqlite:///./data/db/saged_app.db"
    table_prefix: str = "saged_"

class KeywordFinderConfig(BaseModel):
    require: bool = True
    reading_location: str = "default"
    method: str = "embedding_on_wiki"
    keyword_number: int = 7
    hyperlinks_info: List[Any] = []
    llm_info: Dict[str, Any] = {}
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
    database_config: DatabaseConfig = DatabaseConfig()
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

class BenchmarkResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    database_data: Optional[AllDataTiersResponse] = None 