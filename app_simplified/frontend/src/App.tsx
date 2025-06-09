import React from 'react';
import { CssBaseline, ThemeProvider, createTheme, Container, Box } from '@mui/material';
import BenchmarkConfigForm from './components/build_forms';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box sx={{ py: 4 }}>
          <BenchmarkConfigForm />
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App; 