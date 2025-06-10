import React from 'react';
import { DomainBenchmarkConfig } from '../../types/saged_config';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    Typography,
} from '@mui/material';

interface FormValidatorProps {
    config: DomainBenchmarkConfig;
    onClose: () => void;
    open: boolean;
    onConfigUpdate?: (config: DomainBenchmarkConfig) => void;
}

const FormValidator: React.FC<FormValidatorProps> = ({ config, onClose, open, onConfigUpdate }) => {
    const validationErrors = [];
    
    if (!config.domain.trim()) {
        validationErrors.push('Topic name is required');
    }
    if (config.concepts.length === 0) {
        validationErrors.push('At least one concept is required');
    }
    if (!config.shared_config.source_finder.require) {
        validationErrors.push('Please configure your sources (either upload files or use Wikipedia)');
    } else if (config.shared_config.source_finder.method === 'local_files') {
        // Check if any concept has sources configured
        const hasSources = Object.values(config.concept_specified_config).some(
            conceptConfig => conceptConfig.source_finder?.manual_sources && conceptConfig.source_finder.manual_sources.length > 0
        );
        if (!hasSources) {
            validationErrors.push('Please upload and assign sources to at least one concept');
        }
    }
    if (!config.shared_config.prompt_assembler.method) {
        validationErrors.push('Please select a prompt assembly method');
    }

    // Validate and correct keyword finder method
    const validateAndCorrectConfig = () => {
        const updatedConfig = {
            ...config,
            shared_config: {
                ...config.shared_config,
                keyword_finder: {
                    ...config.shared_config.keyword_finder,
                    method: config.shared_config.keyword_finder.require ? 
                        config.shared_config.keyword_finder.method : 
                        'embedding_on_wiki'
                }
            }
        };
        
        if (onConfigUpdate) {
            onConfigUpdate(updatedConfig);
        }
        return updatedConfig;
    };

    const handleClose = () => {
        if (validationErrors.length === 0) {
            validateAndCorrectConfig();
        }
        onClose();
    };

    return (
        <Dialog open={open} onClose={handleClose}>
            <DialogTitle>Configuration Incomplete</DialogTitle>
            <DialogContent>
                <Typography variant="body1" gutterBottom>
                    Please complete the following before submitting:
                </Typography>
                <ul style={{ listStyleType: 'disc', paddingLeft: '20px' }}>
                    {validationErrors.map((error, index) => (
                        <li key={index}>
                            <Typography variant="body2" color="error">
                                {error}
                            </Typography>
                        </li>
                    ))}
                </ul>
            </DialogContent>
            <DialogActions>
                <Button onClick={handleClose} color="primary">
                    Close
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default FormValidator; 