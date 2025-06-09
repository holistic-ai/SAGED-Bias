import React, { useState } from 'react';
import { DomainBenchmarkConfig } from '../../types/saged_config';
import { FormCard } from '../ui/form-card';
import { FormField } from '../ui/form-field';
import { Select } from '../ui/select';
import { FormSwitch } from '../ui/form-switch';
import PromptAssemblerBranching from './PromptAssemblerBranching';

interface PromptAssemblerConfigProps {
    config: DomainBenchmarkConfig;
    onConfigChange: (newConfig: DomainBenchmarkConfig) => void;
}

const PromptAssemblerConfig: React.FC<PromptAssemblerConfigProps> = ({ config, onConfigChange }) => {
    // Local state to control branching UI visibility
    const [showBranching, setShowBranching] = useState(false);

    // Updates shared_config.prompt_assembler.method
    const handleMethodChange = (value: string) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                prompt_assembler: {
                    ...config.shared_config.prompt_assembler,
                    method: value
                }
            }
        });
    };

    // Updates shared_config.prompt_assembler.max_benchmark_length
    const handleMaxBenchmarkLengthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                prompt_assembler: {
                    ...config.shared_config.prompt_assembler,
                    max_benchmark_length: parseInt(e.target.value) || 0
                }
            }
        });
    };

    return (
        <FormCard
            title="Assemble Prompts"
            description="Choose how you want your prompts to be like."
            className="mb-6"
        >
            <div className="space-y-4">
                <div className="space-y-6">
                    {/* Input for shared_config.prompt_assembler.method */}
                    <div className="space-y-2">
                        <label className="text-sm font-medium">Method</label>
                        <div className="text-sm text-gray-500 mb-2">
                            Choose how to process your prompts:
                        </div>
                        <ul className="list-disc pl-5 mb-2 text-sm text-gray-500">
                            <li><strong>Questions:</strong> Converts sentences into questions while maintaining the original meaning and ensuring the concept is included.</li>
                            <li><strong>Split Sentences:</strong> Intelligently splits sentences at natural break points after verbs, creating two related parts that maintain context.</li>
                        </ul>
                        <Select
                            value={config.shared_config.prompt_assembler.method || "questions"}
                            onValueChange={handleMethodChange}
                        >
                            <option value="questions">Questions</option>
                            <option value="split_sentences">Split Sentences</option>
                        </Select>
                    </div>

                    {/* Input for shared_config.prompt_assembler.max_benchmark_length */}
                    <FormField
                        label="Max Benchmark Length"
                        type="number"
                        value={config.shared_config.prompt_assembler.max_benchmark_length}
                        onChange={handleMaxBenchmarkLengthChange}
                        placeholder="Enter maximum benchmark length"
                        description="Set the maximum number of prompts to generate. Leave empty for unlimited."
                    />

                    {/* Controls visibility of branching configuration */}
                    <div className="space-y-4 pt-4 border-t">
                        <FormSwitch
                            label="Enable Concept Branching"
                            checked={showBranching}
                            onChange={(checked) => setShowBranching(checked)}
                            description="Create variations of prompts by replacing concepts with related ones. This helps generate diverse perspectives while maintaining context."
                        />

                        {/* Renders PromptAssemblerBranching component when branching is enabled */}
                        {showBranching && (
                            <div className="pl-6 space-y-4 border-l-2 border-border">
                                <PromptAssemblerBranching
                                    config={config}
                                    onConfigChange={onConfigChange}
                                />
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </FormCard>
    );
};

export default PromptAssemblerConfig;