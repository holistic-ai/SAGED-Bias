import React, { useState } from 'react';
import { DomainBenchmarkConfig } from '../../types/saged_config';
import { FormCard } from '../ui/form-card';
import { Button } from '../ui/button';
import { FormSwitch } from '../ui/form-switch';
import SourceFinderConfig from './SourceFinderConfig';
import FileUploadList from '../ui/file-upload-list';
import { ConceptSelector } from '../ui/concept-selector';
import axios from 'axios';
import { Alert } from '../ui/alert';
import { API_ENDPOINTS } from '../../config/api';

interface SourceSelectionProps {
    config: DomainBenchmarkConfig;
    onConfigChange: (config: DomainBenchmarkConfig) => void;
}

interface SourceWithConcepts {
    file: File;
    concepts: string[];
    uploadedPath?: string;
}

const SourceSelection: React.FC<SourceSelectionProps> = ({ config, onConfigChange }) => {
    // Local state for managing uploaded sources and their concept assignments
    const [sources, setSources] = useState<SourceWithConcepts[]>([]);
    // Local state to control Wikipedia source finder visibility
    const [showSourceFinder, setShowSourceFinder] = useState(false);
    // Local state to track which source is being edited
    const [selectedSource, setSelectedSource] = useState<number | null>(null);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);

    // Handles new file uploads, uploading them to the backend
    const handleFilesChange = async (files: File[]) => {
        setIsUploading(true);
        setUploadError(null);
        
        try {
            // First, delete any removed files from the database
            // Find files to delete (in current sources but not in new files)
            const filesToDelete = sources.filter(source => 
                source.uploadedPath && !files.some(f => f.name === source.file.name)
            );

            // Delete removed files from database
            if (filesToDelete.length > 0) {
                setIsDeleting(true);
                try {
                    await Promise.all(filesToDelete.map(async (source) => {
                        if (source.uploadedPath) {
                            await axios.delete(
                                API_ENDPOINTS.FILES.DELETE(config.domain, source.uploadedPath)
                            );
                        }
                    }));
                } catch (error) {
                    console.error('Error deleting files:', error);
                    setUploadError('Failed to delete some files. Please try again.');
                    return;
                } finally {
                    setIsDeleting(false);
                }
            }

            // Upload new files
            const uploadPromises = files.map(async (file) => {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await axios.post(
                    API_ENDPOINTS.FILES.UPLOAD(config.domain),
                    formData,
                    {
                        headers: {
                            'Content-Type': 'multipart/form-data',
                        },
                    }
                );
                
                return {
                    file,
                    concepts: [...config.concepts],
                    uploadedPath: response.data.data.path
                };
            });

            const uploadedSources = await Promise.all(uploadPromises);
            setSources(uploadedSources);

            // Update config with new sources
            updateSourceConfig(showSourceFinder);
        } catch (error) {
            console.error('Error handling files:', error);
            setUploadError('Failed to handle files. Please check if the backend server is running and try again.');
        } finally {
            setIsUploading(false);
        }
    };

    // Toggles the selected source for editing
    const handleSourceClick = (index: number) => {
        setSelectedSource(selectedSource === index ? null : index);
    };

    // Updates the concepts assigned to a specific source
    const handleSourceConceptsChange = (index: number, concepts: string[]) => {
        setSources(prev => prev.map((source, i) => 
            i === index ? { ...source, concepts } : source
        ));
        // Update config when concepts change
        updateSourceConfig(showSourceFinder);
    };

    // Updates shared_config.source_finder and concept_specified_config with local files or Wikipedia settings
    const updateSourceConfig = (useWikipedia: boolean) => {
        // Create concept-specific source configurations
        const updatedConceptConfig = { ...config.concept_specified_config };
        
        // Group sources by concept
        const conceptSources: Record<string, string[]> = {};
        sources.forEach(source => {
            source.concepts.forEach(concept => {
                if (!conceptSources[concept]) {
                    conceptSources[concept] = [];
                }
                if (source.uploadedPath) {
                    conceptSources[concept].push(source.uploadedPath);
                }
            });
        });

        // Update concept-specific configs only for concepts that have sources
        Object.entries(conceptSources).forEach(([concept, paths]) => {
            // Only update if we have paths for this concept
            if (paths.length > 0) {
                updatedConceptConfig[concept] = {
                    ...updatedConceptConfig[concept], // Preserve existing config
                    source_finder: {
                        ...updatedConceptConfig[concept]?.source_finder, // Preserve existing source finder config
                        manual_sources: paths
                    }
                };
            }
        });

        const method = useWikipedia ? 'wiki' : 'local_files';
        const updatedConfig = {
            ...config,
            concept_specified_config: updatedConceptConfig,
            shared_config: {
                ...config.shared_config,
                source_finder: {
                    ...config.shared_config.source_finder,
                    require: true,
                    method
                },
                scraper: {
                    ...config.shared_config.scraper,
                    method // Sync scraper method with source finder method
                }
            }
        };

        onConfigChange(updatedConfig);
    };

    // Handles Wikipedia toggle
    const handleWikipediaToggle = (checked: boolean) => {
        setShowSourceFinder(checked);
        updateSourceConfig(checked);
    };

    // Updates shared_config.source_finder with local files or Wikipedia settings
    const handleConfirm = () => {
        updateSourceConfig(showSourceFinder);
    };

    return (
        <FormCard
            title="Source Selection"
            description="Upload sources and assign them to concepts"
            className="mb-6"
        >
            <div className="space-y-4">
                {/* File upload section - only shown when not using Wikipedia */}
                <div className="space-y-2">
                    <label className="text-sm font-medium">Upload Sources</label>
                    {!showSourceFinder && (
                        <>
                            <FileUploadList
                                files={sources.map(s => s.file)}
                                onFilesChange={handleFilesChange}
                                disabled={showSourceFinder || isUploading || isDeleting}
                                label="Upload Source Files"
                                emptyMessage="No source files uploaded. Click 'Upload Source Files' to add sources."
                                accept=".txt"
                            />
                            {(isUploading || isDeleting) && (
                                <div className="text-sm text-gray-500">
                                    {isUploading ? 'Uploading files...' : 'Deleting files...'}
                                </div>
                            )}
                            {uploadError && (
                                <Alert variant="destructive" className="mt-2">
                                    {uploadError}
                                </Alert>
                            )}
                        </>
                    )}
                </div>

                {/* Concept assignment section - only shown when files are uploaded */}
                {sources.length > 0 && !showSourceFinder && (
                    <div className="space-y-4">
                        <h3 className="text-sm font-medium">Assign Concepts to Sources</h3>
                        <div className="space-y-2">
                            {sources.map((source, index) => (
                                <div 
                                    key={source.file.name}
                                    className="p-4 border rounded-md hover:bg-gray-50"
                                >
                                    <div 
                                        className="font-medium mb-2 cursor-pointer"
                                        onClick={() => handleSourceClick(index)}
                                    >
                                        {source.file.name}
                                    </div>
                                    {selectedSource === index && (
                                        <div className="pl-4 space-y-2 border-l-2 border-border">
                                            <label className="text-sm text-gray-600">Assigned Concepts</label>
                                            <ConceptSelector
                                                selectedConcepts={source.concepts}
                                                availableConcepts={config.concepts}
                                                onChange={(concepts) => handleSourceConceptsChange(index, concepts)}
                                            />
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Wikipedia source finder toggle and configuration */}
                <div className="space-y-4 pt-4 border-t">
                    <FormSwitch
                        label="Use Wikipedia"
                        checked={showSourceFinder}
                        onChange={handleWikipediaToggle}
                    />

                    {/* Wikipedia source finder configuration - only shown when enabled */}
                    {showSourceFinder && (
                        <div className="pl-6 space-y-4 border-l-2 border-border">
                            <SourceFinderConfig
                                config={config}
                                onConfigChange={onConfigChange}
                            />
                        </div>
                    )}
                </div>

                {/* Confirmation button to update the configuration */}
                <div className="flex justify-end">
                    <Button
                        onClick={handleConfirm}
                        variant="default"
                        className="px-4 py-2"
                    >
                        Confirm Sources
                    </Button>
                </div>
            </div>
        </FormCard>
    );
};

export default SourceSelection; 