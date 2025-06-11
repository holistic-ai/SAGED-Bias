import React from 'react';
import { Box, Container } from '@mui/material';
import BenchmarkConfigFormV2 from '../components/build_forms_v2';

const BenchmarkV2: React.FC = () => {
  return (
    <Container maxWidth="xl" sx={{ height: 'calc(100vh - 64px)', py: 4 }}>
      <Box sx={{ height: '100%' }}>
        <BenchmarkConfigFormV2 />
      </Box>
    </Container>
  );
};

export default BenchmarkV2; 