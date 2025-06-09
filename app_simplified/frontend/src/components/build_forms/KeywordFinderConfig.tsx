import React from 'react';
import { DomainBenchmarkConfig } from '../../types/saged_config';
import { FormField } from '../ui/form-field';

interface KeywordFinderConfigProps {
    config: DomainBenchmarkConfig;
    onConfigChange: (config: DomainBenchmarkConfig) => void;
}

// Map UI values to backend values
const METHOD_MAPPING: Record<string, string> = {
    'wiki_vocabularies': 'embedding_on_wiki',
    'llm_associations': 'llm_inquiries'
};

// Map backend values to UI values
const REVERSE_METHOD_MAPPING: Record<string, string> = {
    'embedding_on_wiki': 'wiki_vocabularies',
    'llm_inquiries': 'llm_associations'
};

const KeywordFinderConfig: React.FC<KeywordFinderConfigProps> = ({ config, onConfigChange }) => {
    // Updates shared_config.keyword_finder.method and sets require to true
    const handleMethodChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const uiValue = e.target.value;
        const backendValue = METHOD_MAPPING[uiValue];
        
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                keyword_finder: {
                    ...config.shared_config.keyword_finder,
                    method: backendValue,
                    require: true
                }
            }
        });
    };

    // Updates shared_config.keyword_finder.keyword_number and sets require to true
    const handleKeywordNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                keyword_finder: {
                    ...config.shared_config.keyword_finder,
                    keyword_number: parseInt(e.target.value) || 0,
                    require: true
                }
            }
        });
    };

    // Updates shared_config.keyword_finder.max_adjustment and sets require to true
    const handleMaxAdjustmentChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                keyword_finder: {
                    ...config.shared_config.keyword_finder,
                    max_adjustment: parseInt(e.target.value) || 0,
                    require: true
                }
            }
        });
    };

    // Updates shared_config.keyword_finder.llm_info.n_run
    const handleNRunChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                keyword_finder: {
                    ...config.shared_config.keyword_finder,
                    llm_info: {
                        ...config.shared_config.keyword_finder.llm_info,
                        n_run: parseInt(e.target.value) || 20
                    },
                    require: true
                }
            }
        });
    };

    // Updates shared_config.keyword_finder.llm_info.n_keywords
    const handleNKeywordsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                keyword_finder: {
                    ...config.shared_config.keyword_finder,
                    llm_info: {
                        ...config.shared_config.keyword_finder.llm_info,
                        n_keywords: parseInt(e.target.value) || 20
                    },
                    require: true
                }
            }
        });
    };

    // Determines which fields to show based on the selected method
    const isWikiVocabularies = config.shared_config.keyword_finder.method === 'embedding_on_wiki';

    return (
        <div className="space-y-4">
            {/* Input for shared_config.keyword_finder.method */}
            <div className="space-y-2">
                <label className="text-sm font-medium">Method</label>
                <select
                    value={REVERSE_METHOD_MAPPING[config.shared_config.keyword_finder.method] || config.shared_config.keyword_finder.method}
                    onChange={handleMethodChange}
                    className="w-full p-2 border rounded-md"
                >
                    <option value="wiki_vocabularies">Wiki Vocabularies</option>
                    <option value="llm_associations">LLM Associations</option>
                </select>
            </div>

            {isWikiVocabularies ? (
                <>
                    {/* Input for shared_config.keyword_finder.keyword_number */}
                    <FormField
                        label="Number of Keywords"
                        type="number"
                        value={config.shared_config.keyword_finder.keyword_number}
                        onChange={handleKeywordNumberChange}
                        placeholder="Enter number of keywords"
                    />
                    {/* Input for shared_config.keyword_finder.max_adjustment */}
                    <FormField
                        label="Candidate Number"
                        type="number"
                        value={config.shared_config.keyword_finder.max_adjustment}
                        onChange={handleMaxAdjustmentChange}
                        placeholder="Enter candidate number"
                    />
                </>
            ) : (
                <>
                    {/* Input for shared_config.keyword_finder.llm_info.n_keywords */}
                    <FormField
                        label="Number of Keywords"
                        type="number"
                        value={config.shared_config.keyword_finder.llm_info?.n_keywords || 20}
                        onChange={handleNKeywordsChange}
                        placeholder="Enter number of keywords"
                    />
                    {/* Input for shared_config.keyword_finder.llm_info.n_run */}
                    <FormField
                        label="Number of Considerations"
                        type="number"
                        value={config.shared_config.keyword_finder.llm_info?.n_run || 20}
                        onChange={handleNRunChange}
                        placeholder="Enter number of considerations"
                    />
                </>
            )}
        </div>
    );
};

export default KeywordFinderConfig; 