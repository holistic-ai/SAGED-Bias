import React, { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";

interface AnalysisResult {
  id: number;
  experiment_id: number;
  feature_results: Array<{
    feature_name: string;
    target_group: string;
    disparity_score: number;
    p_value: number;
    confidence_interval: [number, number];
    effect_size: number;
    significant: boolean;
  }>;
  summary_stats: {
    total_features: number;
    significant_features: number;
    bias_percentage: number;
  };
  created_at: string;
}

interface Experiment {
  id: number;
  name: string;
  status: string;
  created_at: string;
}

const fetchAnalysisResults = async (): Promise<AnalysisResult[]> => {
  const response = await fetch("/api/v1/analysis/results");
  if (!response.ok) {
    throw new Error("Failed to fetch analysis results");
  }
  return response.json();
};

const fetchExperiments = async (): Promise<Experiment[]> => {
  const response = await fetch("/api/v1/experiments");
  if (!response.ok) {
    throw new Error("Failed to fetch experiments");
  }
  return response.json();
};

const Analysis: React.FC = () => {
  const [selectedExperiment, setSelectedExperiment] = useState<number | null>(
    null
  );

  const {
    data: analysisResults,
    isLoading: loadingResults,
    error: resultsError,
  } = useQuery({
    queryKey: ["analysisResults"],
    queryFn: fetchAnalysisResults,
  });

  const {
    data: experiments,
    isLoading: loadingExperiments,
    error: experimentsError,
  } = useQuery({
    queryKey: ["experiments"],
    queryFn: fetchExperiments,
  });

  const filteredResults = analysisResults?.filter(
    (result) =>
      !selectedExperiment || result.experiment_id === selectedExperiment
  );

  const getSignificanceColor = (significant: boolean) => {
    return significant
      ? "bg-red-100 text-red-800 border-red-200"
      : "bg-green-100 text-green-800 border-green-200";
  };

  const getEffectSizeLabel = (effectSize: number) => {
    if (effectSize < 0.2) return "Small";
    if (effectSize < 0.5) return "Medium";
    return "Large";
  };

  if (loadingResults || loadingExperiments) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2">Loading analysis results...</span>
      </div>
    );
  }

  if (resultsError || experimentsError) {
    return (
      <div className="bg-red-50 p-4 rounded-lg border border-red-200">
        <h3 className="text-red-800 font-medium">Error Loading Data</h3>
        <p className="text-red-600">
          {(resultsError as Error)?.message ||
            (experimentsError as Error)?.message}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-2xl font-bold text-gray-900">Bias Analysis</h1>
        <p className="text-gray-600">
          View and analyze bias detection results from experiments
        </p>
      </div>

      {/* Experiment Filter */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <label
          htmlFor="experiment-filter"
          className="block text-sm font-medium text-gray-700 mb-2"
        >
          Filter by Experiment
        </label>
        <select
          id="experiment-filter"
          value={selectedExperiment || ""}
          onChange={(e) =>
            setSelectedExperiment(
              e.target.value ? parseInt(e.target.value) : null
            )
          }
          className="w-full max-w-xs px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        >
          <option value="">All Experiments</option>
          {experiments?.map((exp) => (
            <option key={exp.id} value={exp.id}>
              {exp.name} ({exp.status})
            </option>
          ))}
        </select>
      </div>

      {/* Results Overview */}
      {filteredResults && filteredResults.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">
              Total Experiments
            </h3>
            <p className="text-2xl font-bold text-blue-600">
              {filteredResults.length}
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">
              Average Bias Rate
            </h3>
            <p className="text-2xl font-bold text-orange-600">
              {filteredResults.length > 0
                ? (
                    filteredResults.reduce(
                      (acc, result) =>
                        acc + (result.summary_stats?.bias_percentage || 0),
                      0
                    ) / filteredResults.length
                  ).toFixed(1)
                : 0}
              %
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg border border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">
              Significant Features
            </h3>
            <p className="text-2xl font-bold text-red-600">
              {filteredResults.reduce(
                (acc, result) =>
                  acc + (result.summary_stats?.significant_features || 0),
                0
              )}
            </p>
          </div>
        </div>
      )}

      {/* Analysis Results */}
      <div className="space-y-6">
        {filteredResults?.length === 0 ? (
          <div className="bg-gray-50 p-8 rounded-lg border border-gray-200 text-center">
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No Results Found
            </h3>
            <p className="text-gray-600">
              {selectedExperiment
                ? "No analysis results for the selected experiment."
                : "No analysis results available. Run some experiments to see bias analysis here."}
            </p>
          </div>
        ) : (
          filteredResults?.map((result) => (
            <div
              key={result.id}
              className="bg-white p-6 rounded-lg border border-gray-200"
            >
              {/* Result Header */}
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">
                    Experiment {result.experiment_id} Analysis
                  </h3>
                  <p className="text-sm text-gray-500">
                    {new Date(result.created_at).toLocaleDateString()} at{" "}
                    {new Date(result.created_at).toLocaleTimeString()}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-500">Bias Rate</div>
                  <div className="text-xl font-bold text-orange-600">
                    {result.summary_stats?.bias_percentage?.toFixed(1) || 0}%
                  </div>
                </div>
              </div>

              {/* Summary Stats */}
              <div className="grid grid-cols-3 gap-4 mb-6 p-3 bg-gray-50 rounded">
                <div className="text-center">
                  <div className="text-sm text-gray-500">Total Features</div>
                  <div className="text-lg font-semibold">
                    {result.summary_stats?.total_features || 0}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500">Significant</div>
                  <div className="text-lg font-semibold text-red-600">
                    {result.summary_stats?.significant_features || 0}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-500">Non-Significant</div>
                  <div className="text-lg font-semibold text-green-600">
                    {(result.summary_stats?.total_features || 0) -
                      (result.summary_stats?.significant_features || 0)}
                  </div>
                </div>
              </div>

              {/* Feature Results */}
              <div className="space-y-3">
                <h4 className="font-medium text-gray-900">Feature Analysis</h4>
                {result.feature_results?.map((feature, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded border-2 ${getSignificanceColor(
                      feature.significant
                    )}`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">
                            {feature.feature_name}
                          </span>
                          <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                            {feature.target_group}
                          </span>
                          <span
                            className={`text-sm px-2 py-1 rounded ${
                              feature.significant
                                ? "bg-red-100 text-red-800"
                                : "bg-green-100 text-green-800"
                            }`}
                          >
                            {feature.significant ? "Biased" : "Fair"}
                          </span>
                        </div>
                        <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Disparity:</span>
                            <span className="ml-1 font-medium">
                              {feature.disparity_score?.toFixed(3)}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-500">p-value:</span>
                            <span className="ml-1 font-medium">
                              {feature.p_value?.toFixed(4)}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-500">Effect Size:</span>
                            <span className="ml-1 font-medium">
                              {getEffectSizeLabel(feature.effect_size)}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-500">CI:</span>
                            <span className="ml-1 font-medium">
                              [{feature.confidence_interval?.[0]?.toFixed(2)},{" "}
                              {feature.confidence_interval?.[1]?.toFixed(2)}]
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Analysis;
