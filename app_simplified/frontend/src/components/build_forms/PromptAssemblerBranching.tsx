import React, { useState, useEffect } from 'react';
import { DomainBenchmarkConfig } from '../../types/saged_config';
import { FormSwitch } from '../ui/form-switch';
import { KeywordReplacementPair } from './KeywordReplacementPair';
import { Button } from '../ui/button';
import { Select } from '../ui/select';

interface ConceptPair {
    root: string;
    branch: string;
    keywordReplacements: Array<{ original: string; replacement: string }>;
}

interface PromptAssemblerBranchingProps {
    config: DomainBenchmarkConfig;
    onConfigChange: (config: DomainBenchmarkConfig) => void;
}

const PromptAssemblerBranching: React.FC<PromptAssemblerBranchingProps> = ({ config, onConfigChange }) => {
    // Initialize from config's branching.replacement_descriptor_require
    const [useKeywordHelper, setUseKeywordHelper] = useState<boolean>(
        config.branching_config?.replacement_descriptor_require || false
    );

    // Concept pair management
    const [selectedRoot, setSelectedRoot] = useState<string>('');
    const [branchConcept, setBranchConcept] = useState<string>('');
    const [conceptPairs, setConceptPairs] = useState<ConceptPair[]>([]);
    const [selectedPairIndex, setSelectedPairIndex] = useState<number | null>(null);

    // Adds a new concept pair to the configuration
    const handleAddPair = () => {
        if (selectedRoot && branchConcept) {
            const newPair: ConceptPair = {
                root: selectedRoot,
                branch: branchConcept,
                keywordReplacements: [{
                    original: selectedRoot,
                    replacement: branchConcept
                }]
            };
            setConceptPairs(prev => [...prev, newPair]);
            setBranchConcept('');
        }
    };

    // Removes a concept pair from the configuration
    const handleRemovePair = (index: number) => {
        setConceptPairs(prev => prev.filter((_, i) => i !== index));
        if (selectedPairIndex === index) {
            setSelectedPairIndex(null);
        }
    };

    // Updates keyword replacements for a specific concept pair
    const handleKeywordPairsChange = (index: number, keywordPairs: Array<{ original: string; replacement: string }>) => {
        setConceptPairs(prev => {
            const newPairs = [...prev];
            newPairs[index] = {
                ...newPairs[index],
                keywordReplacements: keywordPairs
            };
            return newPairs;
        });
    };

    // Updates the keyword helper setting
    const handleKeywordHelperChange = (checked: boolean) => {
        setUseKeywordHelper(checked);
    };

    // Updates the selected root concept in local state
    const handleRootConceptChange = (value: string) => {
        setSelectedRoot(value);
    };

    // Updates branching_config whenever keyword helper or pairs change
    useEffect(() => {
        // Create nested dictionary structure
        const replacementDescription = conceptPairs.reduce((acc, pair) => {
            if (!acc[pair.root]) {
                acc[pair.root] = {};
            }
            acc[pair.root][pair.branch] = Object.fromEntries(
                pair.keywordReplacements.map(kr => [kr.original, kr.replacement])
            );
            return acc;
        }, {} as Record<string, Record<string, Record<string, string>>>);

        onConfigChange({
            ...config,
            branching: true,
            branching_config: {
                branching_pairs: "not_all",
                direction: "forward",
                replacement_descriptor_require: useKeywordHelper,
                descriptor_threshold: "Auto",
                descriptor_embedding_model: "paraphrase-Mpnet-base-v2",
                descriptor_distance: "cosine",
                replacement_description: replacementDescription,
                replacement_description_saving: true,
                replacement_description_saving_location: "default",
                counterfactual_baseline: true
            }
        });
    }, [useKeywordHelper, conceptPairs]);

    return (
        <div className="space-y-4">
            {/* Root and Branch concept selection UI */}
            <div className="flex gap-4 items-end">
                <div className="flex-1 space-y-2">
                    <label className="text-sm font-medium">Stem Concept</label>
                    <p className="text-sm text-gray-500 mb-2">
                        Select the main concept that will be used as the base for generating variations. This concept will be replaced with related concepts to create diverse perspectives.
                    </p>
                    <Select
                        value={selectedRoot}
                        onValueChange={handleRootConceptChange}
                    >
                        <option value="">Select stem concept</option>
                        {config.concepts.map((concept) => (
                            <option key={concept} value={concept}>
                                {concept}
                            </option>
                        ))}
                    </Select>
                </div>

                <div className="flex-1 space-y-2">
                    <label className="text-sm font-medium">Branch Concept</label>
                    <p className="text-sm text-gray-500 mb-2">
                        Enter a related concept that will replace the stem concept. This helps create alternative viewpoints while maintaining the original context.
                    </p>
                    <input
                        type="text"
                        value={branchConcept}
                        onChange={(e) => setBranchConcept(e.target.value)}
                        placeholder="Enter branch concept"
                        className="w-full p-2 border rounded-md"
                    />
                </div>

                <Button
                    onClick={handleAddPair}
                    disabled={!selectedRoot || !branchConcept}
                    className="mb-0.5"
                >
                    Add Pair
                </Button>
            </div>

            {/* Concept pairs list and management UI */}
            <div className="space-y-2">
                <label className="text-sm font-medium">Concept Pairs</label>
                <p className="text-sm text-gray-500 mb-2">
                    View and manage your concept pairs. Each pair represents a relationship between concepts that will be used to generate variations of your prompts.
                </p>
                <div className="space-y-2">
                    {conceptPairs.map((pair, index) => (
                        <div key={index} className="space-y-2">
                            <div className="flex items-center gap-2 p-2 bg-gray-50 rounded-md">
                                <span className="flex-1">
                                    <span className="font-medium">{pair.root}</span>
                                    <span className="mx-2">→</span>
                                    <span>{pair.branch}</span>
                                </span>
                                <div className="flex gap-2">
                                    <Button
                                        onClick={() => setSelectedPairIndex(selectedPairIndex === index ? null : index)}
                                        variant="outline"
                                        size="sm"
                                    >
                                        {selectedPairIndex === index ? 'Hide Keywords' : 'Show Keywords'}
                                    </Button>
                                    <Button
                                        onClick={() => handleRemovePair(index)}
                                        variant="ghost"
                                        size="sm"
                                    >
                                        Remove
                                    </Button>
                                </div>
                            </div>
                            
                            {/* Keyword replacements UI for selected pair */}
                            {selectedPairIndex === index && (
                                <div className="pl-4 border-l-2 border-border">
                                    <p className="text-sm text-gray-500 mb-2">
                                        Configure specific keyword replacements between concepts. This helps maintain context and meaning when generating variations.
                                    </p>
                                    <KeywordReplacementPair
                                        pairs={pair.keywordReplacements || []}
                                        onPairsChange={(pairs) => handleKeywordPairsChange(index, pairs)}
                                        stemConcept={pair.root}
                                        branchConcept={pair.branch}
                                    />
                                </div>
                            )}
                        </div>
                    ))}
                    {conceptPairs.length === 0 && (
                        <div className="text-sm text-gray-500 italic">
                            No concept pairs added yet. Add pairs to start generating variations of your prompts.
                        </div>
                    )}
                </div>
            </div>

            {/* Keyword helper toggle UI */}
            <div className="pt-4 border-t">
                <FormSwitch
                    label="Use Keyword Helper"
                    checked={useKeywordHelper}
                    onChange={handleKeywordHelperChange}
                    description="Enable AI assistance for suggesting keyword replacements. This helps identify related terms and maintain context when generating variations."
                />
            </div>
        </div>
    );
};

export default PromptAssemblerBranching; 