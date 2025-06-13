import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  CircularProgress,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { BenchmarkMetadata, BenchmarkMetadataResponse } from '../types/benchmark';

const MetadataRepository: React.FC = () => {
  const [metadata, setMetadata] = useState<BenchmarkMetadataResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMetadata = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/db/benchmark-metadata');
      if (!response.ok) {
        throw new Error('Failed to fetch metadata');
      }
      const data = await response.json();
      setMetadata(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetadata();
  }, []);

  const formatValue = (value: any): React.ReactNode => {
    if (value === null || value === undefined) {
      return '-';
    }
    
    if (typeof value === 'object') {
      return (
        <Box sx={{ maxWidth: 300, overflow: 'auto' }}>
          <pre style={{ margin: 0, whiteSpace: 'pre-wrap', fontSize: '0.875rem' }}>
            {JSON.stringify(value, null, 2)}
          </pre>
        </Box>
      );
    }
    
    return String(value);
  };

  const renderMergedTable = (data: Record<string, any>) => {
    // Parse all string values that might be JSON
    const parsedData: Record<string, any> = {};
    Object.entries(data).forEach(([key, value]) => {
      if (typeof value === 'string') {
        try {
          parsedData[key] = JSON.parse(value);
        } catch (e) {
          parsedData[key] = value;
        }
      } else {
        parsedData[key] = value;
      }
    });

    // Get all unique indices from all data objects
    const allIndices = new Set<number>();
    Object.values(parsedData).forEach(value => {
      if (typeof value === 'object' && value !== null) {
        Object.keys(value).forEach(key => {
          if (!isNaN(Number(key))) {
            allIndices.add(Number(key));
          }
        });
      }
    });

    // Sort indices numerically
    const sortedIndices = Array.from(allIndices).sort((a, b) => a - b);

    return (
      <TableContainer component={Paper} sx={{ mb: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Index</TableCell>
              {Object.keys(parsedData).map((columnName) => (
                <TableCell key={columnName}>{columnName}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedIndices.map((index) => (
              <TableRow key={index}>
                <TableCell>{index}</TableCell>
                {Object.keys(parsedData).map((columnName) => {
                  const columnData = parsedData[columnName];
                  const value = typeof columnData === 'object' && columnData !== null
                    ? columnData[index]
                    : columnData;
                  return (
                    <TableCell key={`${columnName}-${index}`}>
                      {formatValue(value)}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Metadata Repository
        </Typography>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={fetchMetadata}
          disabled={loading}
        >
          Refresh
        </Button>
      </Box>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {error && (
        <Typography color="error" sx={{ my: 2 }}>
          {error}
        </Typography>
      )}

      {metadata && Object.entries(metadata).map(([tableName, records]) => (
        <Box key={tableName} sx={{ mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            {tableName}
          </Typography>
          {records.map((record, recordIndex) => (
            record.data && (
              <Box key={recordIndex}>
                {renderMergedTable(record.data)}
              </Box>
            )
          ))}
        </Box>
      ))}
    </Box>
  );
};

export default MetadataRepository; 