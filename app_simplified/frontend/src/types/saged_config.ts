export interface DatabaseConfig {
    use_database: boolean;      // Used in DomainConfig for database connection settings (default: false)
    database_type: string;      // Used in DomainConfig for database type selection (default: "sqlite")
    database_connection: string; // Used in DomainConfig for connection string (default: "")
    table_prefix: string;       // Used in DomainConfig for table naming (default: "saged_")
}

export interface KeywordFinderConfig {
    require: boolean;           // Used in DomainConfig to enable/disable AI keyword assistance
    reading_location: string;   // Used in KeywordFinderConfig for data source (default: "default")
    method: string;             // Used in KeywordFinderConfig for method selection (default: "embedding_on_wiki")
    keyword_number: number;     // Used in KeywordFinderConfig for number of keywords to find (default: 7)
    hyperlinks_info: any[];     // Used in KeywordFinderConfig for link analysis (default: [])
    llm_info: {                 // Used in KeywordFinderConfig for LLM settings
        n_run: number;          // Number of considerations for LLM method (default: 20)
        n_keywords: number;     // Number of keywords to generate (default: 20)
    };
    max_adjustment: number;     // Used in KeywordFinderConfig for candidate number in wiki method (default: 150)
    embedding_model: string;    // Used in KeywordFinderConfig for embedding selection (default: "paraphrase-Mpnet-base-v2")
    saving: boolean;           // Used in KeywordFinderConfig for save settings (default: true)
    saving_location: string;   // Used in KeywordFinderConfig for save location (default: "default")
    manual_keywords?: string[]; // Used in DomainConfig for storing manually entered keywords
    concept_keywords?: Record<string, Array<{ original: string; replacement: string }>>; // Used in DomainConfig for concept-specific keywords
}

export interface SourceFinderConfig {
    require: boolean;           // Used in SourceSelection to enable/disable source finding
    reading_location: string;   // Used in SourceFinderConfig for data source (default: "default")
    method: string;             // Used in SourceFinderConfig for method selection (default: "wiki")
    local_file?: string;        // Used in SourceSelection for local file input
    scrape_number: number;      // Used in SourceFinderConfig for number of forward links to gather (default: 5)
    saving: boolean;           // Used in SourceFinderConfig for save settings (default: true)
    saving_location: string;   // Used in SourceFinderConfig for save location (default: "default")
    scrape_backlinks: number;   // Used in SourceFinderConfig for number of backward links to gather (default: 0)
    manual_sources?: string[];  // Used in SourceSelection for storing uploaded file names
}

export interface ScraperConfig {
    require: boolean;           // Used in SourceSelection to enable/disable scraping
    reading_location: string;   // Used in SourceSelection for data source (default: "default")
    saving: boolean;           // Used in SourceSelection for save settings (default: true)
    method: string;             // Used in SourceSelection for scraping method (default: "wiki")
    saving_location: string;   // Used in SourceSelection for save location (default: "default")
}

export interface PromptAssemblerConfig {
    require: boolean;           // Used in PromptAssemblerConfig to enable/disable feature
    method: string;             // Used in PromptAssemblerConfig for prompt assembly method (default: "split_sentences")
    generation_function?: any;  // Used in PromptAssemblerConfig for custom generation (default: undefined)
    keyword_list?: string[];    // Used in PromptAssemblerConfig for keyword input
    answer_check: boolean;      // Used in PromptAssemblerConfig for answer validation (default: false)
    saving_location: string;   // Used in PromptAssemblerConfig for save location (default: "default")
    max_benchmark_length: number; // Used in PromptAssemblerConfig for maximum number of prompts (default: 500)
    branching: {               // Used in PromptAssemblerBranching for branching logic
        require: boolean;       // Used in PromptAssemblerConfig to enable/disable branching (default: false)
        method: string;         // Used in PromptAssemblerBranching for branching method (default: "tree")
        branching_pairs: string; // Used in PromptAssemblerBranching for storing concept pairs as JSON (default: "")
        direction: string;      // Used in PromptAssemblerBranching for direction (default: "forward")
        replacement_descriptor_require: boolean; // Used in PromptAssemblerBranching for descriptors (default: false)
        descriptor_threshold: string; // Used in PromptAssemblerBranching for threshold (default: "0.5")
        descriptor_embedding_model: string; // Used in PromptAssemblerBranching for model (default: "paraphrase-Mpnet-base-v2")
        descriptor_distance: string; // Used in PromptAssemblerBranching for distance metric (default: "cosine")
        replacement_description: Record<string, any>; // Used in PromptAssemblerBranching for descriptions (default: {})
        replacement_description_saving: boolean; // Used in PromptAssemblerBranching for save settings (default: true)
        replacement_description_saving_location: string; // Used in PromptAssemblerBranching for save location (default: "default")
        counterfactual_baseline: boolean; // Used in PromptAssemblerBranching for baseline (default: false)
    };
}

export interface ConceptBenchmarkConfig {
    database_config: DatabaseConfig;      // Used in DomainConfig for database settings (all defaults)
    keyword_finder: KeywordFinderConfig;  // Used in KeywordFinderConfig for keyword settings
    source_finder: SourceFinderConfig;    // Used in SourceFinderConfig for source settings
    scraper: ScraperConfig;              // Used in SourceSelection for scraping settings (all defaults)
    prompt_assembler: PromptAssemblerConfig; // Used in PromptAssemblerConfig for prompt settings
}

// Type for the nested replacement description structure
export interface ReplacementDescription {
    [stemConcept: string]: {
        [branchConcept: string]: {
            [originalKeyword: string]: string;
        };
    };
}

export interface BranchingConfig {
    branching_pairs: string;    // Used in PromptAssemblerBranching for pair definitions (default: "")
    direction: string;          // Used in PromptAssemblerBranching for direction (default: "forward")
    source_restriction?: string; // Used in PromptAssemblerBranching for source limits
    replacement_descriptor_require: boolean; // Used in PromptAssemblerBranching for descriptors (default: false)
    descriptor_threshold: string; // Used in PromptAssemblerBranching for threshold (default: "0.5")
    descriptor_embedding_model: string; // Used in PromptAssemblerBranching for model (default: "paraphrase-Mpnet-base-v2")
    descriptor_distance: string; // Used in PromptAssemblerBranching for distance metric (default: "cosine")
    replacement_description: ReplacementDescription; // Used in PromptAssemblerBranching for descriptions (default: {})
    replacement_description_saving: boolean; // Used in PromptAssemblerBranching for save settings (default: true)
    replacement_description_saving_location: string; // Used in PromptAssemblerBranching for save location (default: "default")
    counterfactual_baseline: boolean; // Used in PromptAssemblerBranching for baseline (default: false)
    generation_function?: any;  // Used in PromptAssemblerBranching for custom generation (default: undefined)
}

export interface DomainBenchmarkConfig {
    domain: string;        // Used in DomainConfig for main topic name input
    concepts: string[];         // Used in DomainConfig for concept list management
    branching: boolean;         // Used in PromptAssemblerBranching to enable/disable (default: false)
    branching_config?: BranchingConfig; // Used in PromptAssemblerBranching for settings (all defaults)
    shared_config: ConceptBenchmarkConfig; // Used across all config components
    concept_specified_config: Record<string, {  // Used in DomainConfig for concept-specific settings
        keyword_finder?: {                      // Each concept can have its own keyword finder config
            manual_keywords?: string[];         // Manual keywords specific to this concept
        };
    }>;                                        // Default: {}
    saving: boolean;           // Used in DomainConfig for save settings (default: true)
    saving_location: string;   // Used in DomainConfig for save location (default: "default")
    database_config: DatabaseConfig; // Used in DomainConfig for database settings (all defaults)
}

export interface BenchmarkResponse {
    status: string;
    message: string;
    data?: Record<string, any>;
    database_data?: {
        keywords?: any;
        source_finder?: any;
        scraped_sentences?: any;
        split_sentences?: any;
        questions?: any;
        replacement_description?: any;
    };
}

// Default configuration with all necessary settings
export const defaultConfig: DomainBenchmarkConfig = {
    domain: '',
    concepts: [],
    branching: false,
    shared_config: {
        database_config: {
            use_database: false,
            database_type: "sqlite",
            database_connection: "",
            table_prefix: "saged_"
        },
        keyword_finder: {
            require: false,
            reading_location: "default",
            method: "embedding_on_wiki",
            keyword_number: 7,
            hyperlinks_info: [],
            llm_info: {
                n_run: 20,
                n_keywords: 20
            },
            max_adjustment: 150,
            embedding_model: "paraphrase-Mpnet-base-v2",
            saving: true,
            saving_location: "default"
        },
        source_finder: {
            require: false,
            reading_location: "default",
            method: "wiki",
            scrape_number: 5,
            saving: true,
            saving_location: "default",
            scrape_backlinks: 0
        },
        scraper: {
            require: false,
            reading_location: "default",
            saving: true,
            method: "wiki",
            saving_location: "default"
        },
        prompt_assembler: {
            require: false,
            method: "split_sentences",
            answer_check: false,
            saving_location: "default",
            max_benchmark_length: 500,
            branching: {
                require: false,
                method: "tree",
                branching_pairs: "",
                direction: "forward",
                replacement_descriptor_require: false,
                descriptor_threshold: "0.5",
                descriptor_embedding_model: "paraphrase-Mpnet-base-v2",
                descriptor_distance: "cosine",
                replacement_description: {},
                replacement_description_saving: true,
                replacement_description_saving_location: "default",
                counterfactual_baseline: false
            }
        }
    },
    concept_specified_config: {},
    saving: true,
    saving_location: "default",
    database_config: {
        use_database: false,
        database_type: "sqlite",
        database_connection: "",
        table_prefix: "saged_"
    },
    branching_config: {
        branching_pairs: "",
        direction: "forward",
        replacement_descriptor_require: false,
        descriptor_threshold: "Auto",
        descriptor_embedding_model: "paraphrase-Mpnet-base-v2",
        descriptor_distance: "cosine",
        replacement_description: {},
        replacement_description_saving: true,
        replacement_description_saving_location: "default",
        counterfactual_baseline: true
    }
}; 