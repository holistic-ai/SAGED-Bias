import React, { useState } from 'react';
import { DomainBenchmarkConfig, BenchmarkResponse, defaultConfig } from '../types/saged_config';
import {
    Box,
    Button,
    Typography,
    Alert,
} from '@mui/material';
import DomainConfig from './build_forms/DomainConfig';
import PromptAssemblerConfig from './build_forms/PromptAssemblerConfig';
import SourceSelection from './build_forms/SourceSelection';
import FormValidator from './build_forms/FormValidator';
import BenchmarkResults from './BenchmarkResults';

const BenchmarkConfigForm: React.FC = () => {
    const [config, setConfig] = useState<DomainBenchmarkConfig>(defaultConfig);
    const [response, setResponse] = useState<BenchmarkResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showValidation, setShowValidation] = useState(false);

    const validateForm = (): boolean => {
        if (!config.domain.trim() || 
            config.concepts.length === 0 || 
            !config.shared_config.source_finder.require ||
            !config.shared_config.prompt_assembler.method) {
            setShowValidation(true);
            return false;
        }
        return true;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (!validateForm()) {
            return;
        }

        // Log what is being sent to the backend
        console.log('Sending data to backend:', {
            url: 'http://localhost:8000/benchmark/build',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: config
        });

        try {
            const response = await fetch('http://localhost:8000/benchmark/build', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                if (errorData?.detail) {
                    throw new Error(`Server error: ${errorData.detail}`);
                }
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Response from backend:', data);
            setResponse(data);
            setError(null);
        } catch (err) {
            console.error('Error sending configuration:', err);
            let errorMessage = 'An error occurred while connecting to the server.';
            
            if (err instanceof Error) {
                if (err.message.includes('Failed to fetch')) {
                    errorMessage = 'Unable to connect to the server. Please check if the backend service is running at http://localhost:8000.';
                } else if (err.message.includes('Server error:')) {
                    errorMessage = err.message;
                } else {
                    errorMessage = `Error: ${err.message}`;
                }
            }
            
            setError(errorMessage);
            setResponse(null);
        }
    };

    return (
        <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
            <Typography variant="h4" gutterBottom>
                SAGED Benchmark Configuration
            </Typography>

            <div className="space-y-6">
                {/* Domain Configuration - Handles domain name, concepts, and keywords */}
                <DomainConfig 
                    config={config} 
                    onConfigChange={setConfig} 
                />

                {/* Source Configuration - Handles source selection and settings */}
                <SourceSelection
                    config={config}
                    onConfigChange={setConfig}
                />

                {/* Prompt Assembly Configuration - Handles prompt generation and branching */}
                <PromptAssemblerConfig 
                    config={config} 
                    onConfigChange={setConfig} 
                />

                {/* Error and Response Messages */}
                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

                {response && (
                    <Alert severity={response.status === 'success' ? 'success' : 'info'} sx={{ mb: 2 }}>
                        {response.message}
                    </Alert>
                )}

                {/* Submit Button */}
                <Button
                    onClick={handleSubmit}
                    variant="contained"
                    color="primary"
                    size="large"
                    fullWidth
                >
                    Build Benchmark
                </Button>

                {/* Display Results */}
                {response && <BenchmarkResults response={response} />}
            </div>

            {/* Form Validation Dialog */}
            <FormValidator
                config={config}
                open={showValidation}
                onClose={() => setShowValidation(false)}
                onConfigUpdate={setConfig}
            />
        </Box>
    );
};

export default BenchmarkConfigForm; 