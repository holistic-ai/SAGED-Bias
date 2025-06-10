import React, { useState } from 'react';
import { DomainBenchmarkConfig } from '../../types/saged_config';
import { FormCard } from '../ui/form-card';
import { FormField } from '../ui/form-field';
import { ConceptList } from '../ui/concept-list';
import { Button } from '../ui/button';
import { Alert } from '../ui/alert';
import { FormSwitch } from '../ui/form-switch';
import KeywordFinderConfig from './KeywordFinderConfig';

interface DomainConfigProps {
    config: DomainBenchmarkConfig;
    onConfigChange: (config: DomainBenchmarkConfig) => void;
}

const DomainConfig: React.FC<DomainConfigProps> = ({ config, onConfigChange }) => {
    const [tempConfig, setTempConfig] = useState<DomainBenchmarkConfig>(config);
    const [isEditing, setIsEditing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedConcept, setSelectedConcept] = useState<string | null>(null);
    const [showKeywordFinder, setShowKeywordFinder] = useState(false);
    const [conceptKeywords, setConceptKeywords] = useState<Record<string, string[]>>(() => {
        // Initialize with existing keywords or empty arrays
        const initialKeywords: Record<string, string[]> = {};
        config.concepts.forEach(concept => {
            initialKeywords[concept] = config.concept_specified_config[concept]?.keyword_finder?.manual_keywords || [concept];
        });
        return initialKeywords;
    });

    // Updates domain in config
    const handleDomainNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setTempConfig({
            ...tempConfig,
            domain: e.target.value
        });
    };

    // Updates concepts array and initializes keywords for new concepts
    const handleConceptsChange = (concepts: string[]) => {
        setTempConfig({
            ...tempConfig,
            concepts
        });
        // Update concept keywords for new concepts
        const newConceptKeywords: Record<string, string[]> = {};
        concepts.forEach(concept => {
            newConceptKeywords[concept] = conceptKeywords[concept] || [concept];
        });
        setConceptKeywords(newConceptKeywords);
    };

    const handleEdit = () => {
        setIsEditing(true);
        setTempConfig(config);
        setError(null);
    };

    const handleCancel = () => {
        setIsEditing(false);
        setTempConfig(config);
        setError(null);
        setSelectedConcept(null);
    };

    // Updates multiple config values when confirming changes:
    // - domain
    // - concepts
    // - shared_config.keyword_finder.require (only when AI assistant is enabled)
    // - concept_specified_config[concept].keyword_finder.manual_keywords
    // - shared_config.keyword_finder.method (only when AI assistant is enabled)
    const handleConfirm = () => {
        // Validate the configuration
        if (!tempConfig.domain.trim()) {
            setError('Topic name is required');
            return;
        }
        if (tempConfig.concepts.length === 0) {
            setError('At least one concept is required');
            return;
        }

        // Update concept-specific configs while preserving existing configurations
        const updatedConceptConfig = { ...tempConfig.concept_specified_config };
        Object.entries(conceptKeywords).forEach(([concept, keywords]) => {
            updatedConceptConfig[concept] = {
                ...updatedConceptConfig[concept], // Preserve existing config
                keyword_finder: {
                    ...updatedConceptConfig[concept]?.keyword_finder, // Preserve existing keyword finder config
                    manual_keywords: keywords
                }
            };
        });

        const updatedConfig = {
            ...tempConfig,
            concept_specified_config: updatedConceptConfig,
            shared_config: {
                ...tempConfig.shared_config,
                keyword_finder: {
                    ...tempConfig.shared_config.keyword_finder,
                    require: showKeywordFinder,
                    ...(showKeywordFinder ? {
                        method: tempConfig.shared_config.keyword_finder.method
                    } : {
                        method: 'embedding_on_wiki'  // Default method when not using AI assistant
                    })
                }
            }
        };

        onConfigChange(updatedConfig);
        setIsEditing(false);
        setError(null);
        setSelectedConcept(null);
    };

    const handleConceptClick = (concept: string) => {
        setSelectedConcept(selectedConcept === concept ? null : concept);
    };

    // Updates keywords for a specific concept in conceptKeywords state
    const handleKeywordsChange = (keywords: string[]) => {
        if (selectedConcept) {
            setConceptKeywords(prev => ({
                ...prev,
                [selectedConcept]: keywords
            }));
        }
    };

    return (
        <FormCard
            title="Set Up Your Topic"
            description="Define the main topic and related concepts you want to explore"
            className="mb-6"
        >
            <div className="space-y-4">
                {error && (
                    <Alert variant="destructive" className="mb-4">
                        {error}
                    </Alert>
                )}

                {/* Input for domain */}
                <FormField
                    label="Topic Name"
                    value={isEditing ? tempConfig.domain : config.domain}
                    onChange={handleDomainNameChange}
                    placeholder="Enter your main topic"
                    required
                    disabled={!isEditing}
                    description="The main subject or area you want to explore (e.g., 'Artificial Intelligence', 'Climate Change')"
                />

                <div className="space-y-4">
                    {/* Input for concepts array */}
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Key Concepts</label>
                        <p className="text-sm text-gray-500 mb-2">
                            Add the main ideas or themes you want to explore within your topic. These will be used to generate relevant content and questions.
                        </p>
                        <ConceptList
                            concepts={isEditing ? tempConfig.concepts : config.concepts}
                            onChange={handleConceptsChange}
                            disabled={!isEditing}
                            onConceptClick={handleConceptClick}
                            selectedConcept={selectedConcept}
                        />
                    </div>

                    {/* Input for concept-specific keywords */}
                    {selectedConcept && (
                        <div className="pl-6 space-y-4 border-l-2 border-border">
                            <div className="space-y-2">
                                <label className="text-sm font-medium">Related Terms for {selectedConcept}</label>
                                <p className="text-sm text-gray-500 mb-2">
                                    Add specific terms or phrases related to this concept. These will help generate more focused and relevant content.
                                </p>
                                <ConceptList
                                    concepts={conceptKeywords[selectedConcept] || []}
                                    onChange={handleKeywordsChange}
                                />
                            </div>
                        </div>
                    )}

                    {/* Toggle for keyword finder and its configuration */}
                    <div className="pt-4 border-t">
                        <FormSwitch
                            label="Use AI Keyword Assistant"
                            checked={showKeywordFinder}
                            onChange={(checked) => setShowKeywordFinder(checked)}
                            description="Let AI help you discover relevant terms and phrases for your concepts. This can help expand your search and find more comprehensive content."
                        />

                        {showKeywordFinder && (
                            <div className="pl-6 mt-4">
                                <KeywordFinderConfig
                                    config={tempConfig}
                                    onConfigChange={setTempConfig}
                                />
                            </div>
                        )}
                    </div>
                </div>

                <div className="flex justify-end space-x-2 mt-4">
                    {!isEditing ? (
                        <Button onClick={handleEdit} variant="outline">
                            Edit Settings
                        </Button>
                    ) : (
                        <>
                            <Button onClick={handleCancel} variant="outline">
                                Cancel
                            </Button>
                            <Button onClick={handleConfirm} variant="default">
                                Save Changes
                            </Button>
                        </>
                    )}
                </div>
            </div>
        </FormCard>
    );
};

export default DomainConfig; 