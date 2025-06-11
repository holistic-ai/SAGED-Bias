import React from 'react';
import { Box, Container } from '@mui/material';
import BenchmarkConfigForm from '../components/build_forms';

const BenchmarkConfig: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <BenchmarkConfigForm />
      </Box>
    </Container>
  );
};

export default BenchmarkConfig; 