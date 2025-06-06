import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import QuickAnalysis from "./pages/QuickAnalysis";
import QuickAnalysisComparison from "./pages/QuickAnalysisComparison";
import Benchmarks from "./pages/Benchmarks";
import Experiments from "./pages/Experiments";
import Analysis from "./pages/Analysis";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/quick-analysis" element={<QuickAnalysis />} />
            <Route
              path="/model-comparison"
              element={<QuickAnalysisComparison />}
            />
            <Route path="/benchmarks" element={<Benchmarks />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/analysis" element={<Analysis />} />
          </Routes>
        </Layout>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
