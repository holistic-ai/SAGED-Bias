import React, { useState } from 'react';
import { DomainBenchmarkConfig, BenchmarkResponse, defaultConfig } from '../types/saged_config';
import {
    Box,
    Button,
    Typography,
    Alert,
    Paper,
    Divider,
    Tabs,
    Tab,
} from '@mui/material';
import DomainConfig from './build_forms/DomainConfig';
import PromptAssemblerConfig from './build_forms/PromptAssemblerConfig';
import SourceSelection from './build_forms/SourceSelection';
import FormValidator from './build_forms/FormValidator';
import BenchmarkResults from './BenchmarkResults';
import { API_ENDPOINTS } from '../config/api';

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;

    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`tabpanel-${index}`}
            aria-labelledby={`tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ py: 2 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

const BenchmarkConfigFormV2: React.FC = () => {
    const [config, setConfig] = useState<DomainBenchmarkConfig>(defaultConfig);
    const [response, setResponse] = useState<BenchmarkResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showValidation, setShowValidation] = useState(false);
    const [tabValue, setTabValue] = useState(0);

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

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

        try {
            const response = await fetch(API_ENDPOINTS.BENCHMARK.BUILD, {
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
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Main Content */}
            <Paper sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                {/* Navigation Tabs */}
                <Tabs 
                    value={tabValue} 
                    onChange={handleTabChange}
                    variant="scrollable"
                    scrollButtons="auto"
                    sx={{ borderBottom: 1, borderColor: 'divider' }}
                >
                    <Tab label="Domain Config" />
                    <Tab label="Source Config" />
                    <Tab label="Prompt Config" />
                    <Tab label="Results" />
                </Tabs>

                {/* Tab Content */}
                <Box sx={{ flex: 1, overflow: 'auto', p: 3 }}>
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

                    {/* Configuration Tabs */}
                    <TabPanel value={tabValue} index={0}>
                        <DomainConfig 
                            config={config} 
                            onConfigChange={setConfig} 
                        />
                    </TabPanel>
                    <TabPanel value={tabValue} index={1}>
                        <SourceSelection
                            config={config}
                            onConfigChange={setConfig}
                        />
                    </TabPanel>
                    <TabPanel value={tabValue} index={2}>
                        <PromptAssemblerConfig 
                            config={config} 
                            onConfigChange={setConfig} 
                        />
                    </TabPanel>

                    {/* Results Tab */}
                    <TabPanel value={tabValue} index={3}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                            {/* Build Button Section */}
                            <Paper 
                                sx={{ 
                                    p: 2, 
                                    bgcolor: 'background.default',
                                    border: '1px dashed',
                                    borderColor: 'divider',
                                    textAlign: 'center'
                                }}
                            >
                                <Button
                                    onClick={handleSubmit}
                                    variant="contained"
                                    color="primary"
                                    size="large"
                                    sx={{ 
                                        minWidth: 200,
                                        py: 1.5,
                                        px: 4
                                    }}
                                >
                                    Build Benchmark
                                </Button>
                            </Paper>

                            {/* Results Section */}
                            {response ? (
                                <BenchmarkResults response={response} />
                            ) : (
                                <Paper sx={{ p: 3, textAlign: 'center' }}>
                                    <Typography color="text.secondary">
                                        No results yet. Configure your settings and run the benchmark to see results.
                                    </Typography>
                                </Paper>
                            )}
                        </Box>
                    </TabPanel>
                </Box>
            </Paper>

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

export default BenchmarkConfigFormV2; 