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

interface BenchmarkResultsProps {
    response: BenchmarkResponse;
}

const BenchmarkResults: React.FC<BenchmarkResultsProps> = ({ response }) => {
    const renderDataFrame = (data: any, title: string) => {
        if (!data || Object.keys(data).length === 0) return null;

        // Handle DataFrame structure
        const columns = data.columns || [];
        const rows = data.data || [];
        const index = data.index || [];

        return (
            <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                    {title}
                </Typography>
                <TableContainer component={Paper}>
                    <Table>
                        <TableHead>
                            <TableRow>
                                <TableCell>Index</TableCell>
                                {columns.map((column: string) => (
                                    <TableCell key={column}>{column}</TableCell>
                                ))}
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {rows.map((row: any, i: number) => (
                                <TableRow key={i}>
                                    <TableCell>{index[i] || i}</TableCell>
                                    {columns.map((column: string) => (
                                        <TableCell key={`${i}-${column}`}>
                                            {row[column] !== undefined ? String(row[column]) : ''}
                                        </TableCell>
                                    ))}
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Box>
        );
    };

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

                    {response.data.data && renderDataFrame(response.data.data, 'Benchmark Data')}
                    
                    {response.data.configuration && renderConfigTable(response.data.configuration, 'Configuration')}
                    
                    {response.data.database_config && renderConfigTable(response.data.database_config, 'Database Configuration')}
                    
                    {response.data.table_names && renderConfigTable(response.data.table_names, 'Table Names')}
                </>
            )}

            {response.database_data && (
                <Box sx={{ mt: 2 }}>
                    {response.database_data.keywords && 
                        renderDataFrame(response.database_data.keywords, 'Keywords')}
                    
                    {response.database_data.source_finder && 
                        renderDataFrame(response.database_data.source_finder, 'Source Finder Results')}
                    
                    {response.database_data.scraped_sentences && 
                        renderDataFrame(response.database_data.scraped_sentences, 'Scraped Sentences')}
                    
                    {response.database_data.split_sentences && 
                        renderDataFrame(response.database_data.split_sentences, 'Split Sentences')}
                    
                    {response.database_data.questions && 
                        renderDataFrame(response.database_data.questions, 'Generated Questions')}
                    
                    {response.database_data.replacement_description && 
                        renderDataFrame(response.database_data.replacement_description, 'Replacement Descriptions')}
                </Box>
            )}
        </Box>
    );
};

export default BenchmarkResults; 