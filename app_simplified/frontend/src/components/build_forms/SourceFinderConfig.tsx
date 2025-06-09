import React from 'react';
import { DomainBenchmarkConfig } from '../../types/saged_config';
import { FormCard } from '../ui/form-card';
import { FormField } from '../ui/form-field';

interface SourceFinderConfigProps {
    config: DomainBenchmarkConfig;
    onConfigChange: (config: DomainBenchmarkConfig) => void;
}

const SourceFinderConfig: React.FC<SourceFinderConfigProps> = ({ config, onConfigChange }) => {
    // Updates shared_config.source_finder.scrape_number and sets method to 'wiki'
    const handleForelinkNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                source_finder: {
                    ...config.shared_config.source_finder,
                    method: 'wiki',
                    scrape_number: parseInt(e.target.value) || 0
                }
            }
        });
    };

    // Updates shared_config.source_finder.scrape_backlinks and sets method to 'wiki'
    const handleBacklinkNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        onConfigChange({
            ...config,
            shared_config: {
                ...config.shared_config,
                source_finder: {
                    ...config.shared_config.source_finder,
                    method: 'wiki',
                    scrape_backlinks: parseInt(e.target.value) || 0
                }
            }
        });
    };

    return (
        <FormCard
            title="Find Sources"
            description="Configure how to gather information from Wikipedia"
            className="mb-6"
        >
            <div className="space-y-4">
                {/* Input for shared_config.source_finder.scrape_number */}
                <FormField
                    label="Forward Links"
                    type="number"
                    value={config.shared_config.source_finder.scrape_number}
                    onChange={handleForelinkNumberChange}
                    placeholder="Enter number of forward links to gather"
                    description="Number of Wikipedia pages to explore from the main article. These are pages that the main article links to."
                />

                {/* Input for shared_config.source_finder.scrape_backlinks */}
                <FormField
                    label="Backward Links"
                    type="number"
                    value={config.shared_config.source_finder.scrape_backlinks}
                    onChange={handleBacklinkNumberChange}
                    placeholder="Enter number of backward links to gather"
                    description="Number of Wikipedia pages that link back to the main article. These help find related content and context."
                />
            </div>
        </FormCard>
    );
};

export default SourceFinderConfig; 