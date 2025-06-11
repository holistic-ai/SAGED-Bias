import React from 'react';
import { CssBaseline, ThemeProvider, createTheme, Box } from '@mui/material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import BenchmarkConfig from './pages/BenchmarkConfig';
import BenchmarkV2 from './pages/BenchmarkV2';
import Navigation from './components/Navigation';

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
      <Router>
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <Navigation />
          <Box component="main" sx={{ flexGrow: 1 }}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/benchmark" element={<BenchmarkConfig />} />
              <Route path="/benchmark-v2" element={<BenchmarkV2 />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App; 