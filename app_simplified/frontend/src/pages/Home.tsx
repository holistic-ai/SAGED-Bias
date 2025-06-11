import React from 'react';
import { Box, Container, Typography, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const Home: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 8, textAlign: 'center' }}>
        <Typography variant="h2" component="h1" gutterBottom>
          Welcome to SAGED
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          A powerful tool for benchmarking and analyzing AI models
        </Typography>
        <Box sx={{ mt: 4 }}>
          <Button
            variant="contained"
            size="large"
            onClick={() => navigate('/benchmark')}
            sx={{ mr: 2 }}
          >
            Start Benchmark
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export default Home; 