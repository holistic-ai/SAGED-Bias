import React from 'react';
import { BenchmarkResponse } from '../types/saged_config';
import {
    Box,
    Paper,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
} from '@mui/material';
import { DataTable } from './ui/data-table';

interface BenchmarkResultsProps {
    response: BenchmarkResponse;
}

const BenchmarkResults: React.FC<BenchmarkResultsProps> = ({ response }) => {
    const renderConfigTable = (data: any, title: string) => {
        if (!data || Object.keys(data).length === 0) return null;

        return (
            <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                    {title}
                </Typography>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Key</TableCell>
                                <TableCell>Value</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {Object.entries(data).map(([key, value]) => (
                                <TableRow key={key}>
                                    <TableCell>{key}</TableCell>
                                    <TableCell>
                                        {typeof value === 'object' 
                                            ? JSON.stringify(value, null, 2)
                                            : String(value)}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Box>
        );
    };

    return (
        <Box sx={{ mt: 4 }}>
            <Typography variant="h5" gutterBottom>
                Benchmark Results
            </Typography>

            {response.status && (
                <Typography variant="subtitle1" color="primary" gutterBottom>
                    Status: {response.status}
                </Typography>
            )}

            {response.message && (
                <Typography variant="body1" gutterBottom>
                    {response.message}
                </Typography>
            )}

            {response.data && (
                <>
                    {response.data.domain && (
                        <Typography variant="subtitle1" gutterBottom>
                            Domain: {response.data.domain}
                        </Typography>
                    )}
                    
                    {response.data.time_stamp && (
                        <Typography variant="subtitle2" gutterBottom>
                            Generated at: {response.data.time_stamp}
                        </Typography>
                    )}

                    {response.data.data && (
                        <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Benchmark Data
                            </Typography>
                            <DataTable data={response.data.data} />
                        </Box>
                    )}
                    
                    {response.data.configuration && renderConfigTable(response.data.configuration, 'Configuration')}
                    
                    {response.data.database_config && renderConfigTable(response.data.database_config, 'Database Configuration')}
                    
                    {response.data.table_names && renderConfigTable(response.data.table_names, 'Table Names')}
                </>
            )}

            {response.database_data && (
                <Box sx={{ mt: 2 }}>
                    {response.database_data.keywords && (
                        <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Keywords
                            </Typography>
                            <DataTable data={response.database_data.keywords} />
                        </Box>
                    )}
                    
                    {response.database_data.source_finder && (
                        <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Source Finder Results
                            </Typography>
                            <DataTable data={response.database_data.source_finder} />
                        </Box>
                    )}
                    
                    {response.database_data.scraped_sentences && (
                        <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Scraped Sentences
                            </Typography>
                            <DataTable data={response.database_data.scraped_sentences} />
                        </Box>
                    )}
                    
                    {response.database_data.split_sentences && (
                        <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Split Sentences
                            </Typography>
                            <DataTable data={response.database_data.split_sentences} />
                        </Box>
                    )}
                    
                    {response.database_data.questions && (
                        <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Generated Questions
                            </Typography>
                            <DataTable data={response.database_data.questions} />
                        </Box>
                    )}
                    
                    {response.database_data.replacement_description && (
                        <Box sx={{ mt: 3 }}>
                            <Typography variant="h6" gutterBottom>
                                Replacement Descriptions
                            </Typography>
                            <DataTable data={response.database_data.replacement_description} />
                        </Box>
                    )}
                </Box>
            )}
        </Box>
    );
};

export default BenchmarkResults; 