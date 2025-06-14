import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { BenchmarkMetadataResponse } from '../types/benchmark';
import { CollapsibleCard } from '../components/ui/card';
import { DataTable } from '../components/ui/data-table';

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
              <CollapsibleCard
                key={recordIndex}
                title={`Record ${recordIndex + 1}`}
                description={`Table: ${tableName}`}
                defaultCollapsed={true}
                className="mb-4"
              >
                <DataTable data={record.data} />
              </CollapsibleCard>
            )
          ))}
        </Box>
      ))}
    </Box>
  );
};

export default MetadataRepository; 