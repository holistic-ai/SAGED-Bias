import React, { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";

interface QuickAnalysisRequest {
  topic: string;
  bias_category: string;
  models_to_test: string[];
  include_baseline: boolean;
}

interface ModelResult {
  model_name: string;
  test_prompts: string[];
  model_responses: string[];
  keywords_found: string[];
  sentiment_analysis: {
    average_sentiment: number;
    sentiment_distribution: { [key: string]: number };
  };
  bias_indicators: {
    keyword: string;
    sentiment_score: number;
    bias_detected: boolean;
  }[];
  summary: {
    bias_detected: boolean;
    confidence_score: number;
    recommendation: string;
  };
}

interface QuickAnalysisResult {
  analysis_id: string;
  topic: string;
  bias_category: string;
  model_results: ModelResult[];
  baseline_result?: ModelResult;
  comparative_analysis: {
    most_biased_model: string;
    least_biased_model: string;
    bias_score_differences: { [key: string]: number };
  };
  processing_time: number;
}

const runQuickAnalysis = async (
  request: QuickAnalysisRequest
): Promise<QuickAnalysisResult> => {
  const response = await fetch("/api/v1/saged/quick-analysis", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error("Failed to run analysis");
  }

  return response.json();
};

const QuickAnalysisComparison: React.FC = () => {
  const [topic, setTopic] = useState("");
  const [biasCategory, setBiasCategory] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([
    "gpt-4.1",
    "o4-mini",
  ]);
  const [includeBaseline, setIncludeBaseline] = useState(true);
  const [result, setResult] = useState<QuickAnalysisResult | null>(null);

  // Latest OpenAI models as of 2025
  const availableModels = [
    {
      id: "gpt-4.1",
      name: "GPT-4.1 (OpenAI Latest 2025) - 1M context, enhanced reasoning",
    },
    {
      id: "gpt-4.1-nano",
      name: "GPT-4.1 Nano (OpenAI Fastest) - Optimized for speed",
    },
    {
      id: "gpt-4.1-mini",
      name: "GPT-4.1 Mini (OpenAI Balanced) - Performance/cost balanced",
    },
    {
      id: "o4-mini",
      name: "o4-mini (OpenAI Reasoning 2025) - Latest reasoning model",
    },
    {
      id: "o3",
      name: "o3 (OpenAI Advanced Reasoning) - Enhanced problem solving",
    },
    {
      id: "o3-mini",
      name: "o3-mini (OpenAI Small Reasoning) - Efficient reasoning",
    },
    { id: "gpt-4o", name: "GPT-4o (OpenAI 2024) - Multimodal capabilities" },
    {
      id: "gpt-4o-mini",
      name: "GPT-4o Mini (OpenAI) - Smaller, faster, cost-effective",
    },
    { id: "gpt-4", name: "GPT-4 (OpenAI) - Previous generation" },
    { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo (OpenAI) - Cost-effective" },
    { id: "claude-3-opus", name: "Claude 3 Opus (Anthropic)" },
    { id: "claude-3-sonnet", name: "Claude 3 Sonnet (Anthropic)" },
    { id: "gemini-pro", name: "Gemini Pro (Google)" },
  ];

  const analysisMutation = useMutation({
    mutationFn: runQuickAnalysis,
    onSuccess: (data) => {
      setResult(data);
    },
    onError: (error) => {
      console.error("Analysis failed:", error);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!topic.trim() || !biasCategory.trim()) {
      alert("Please fill in both topic and bias category");
      return;
    }

    if (selectedModels.length === 0) {
      alert("Please select at least one model to test");
      return;
    }

    analysisMutation.mutate({
      topic: topic.trim(),
      bias_category: biasCategory.trim(),
      models_to_test: selectedModels,
      include_baseline: includeBaseline,
    });
  };

  const resetForm = () => {
    setTopic("");
    setBiasCategory("");
    setSelectedModels(["gpt-4.1", "o4-mini"]);
    setIncludeBaseline(true);
    setResult(null);
  };

  const handleModelToggle = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  };

  const exampleTopics = [
    { topic: "software engineering", category: "gender" },
    { topic: "nursing profession", category: "gender" },
    { topic: "leadership roles", category: "nationality" },
    { topic: "scientific research", category: "race" },
    { topic: "teaching profession", category: "age" },
  ];

  const handleExampleClick = (example: { topic: string; category: string }) => {
    setTopic(example.topic);
    setBiasCategory(example.category);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 p-6">
      {/* Header */}
      <div className="border-b border-gray-200 pb-4">
        <h1 className="text-2xl font-bold text-gray-900">
          AI Model Bias Comparison
        </h1>
        <p className="text-gray-600">
          Compare bias patterns across multiple AI models with baseline analysis
        </p>
        <div className="mt-2 text-sm text-blue-600 bg-blue-50 p-3 rounded-lg">
          <strong>How it works:</strong> Select multiple AI models, enter a
          topic and bias category. We'll test all models with the same prompts
          and provide a detailed comparison of bias patterns.
        </div>
      </div>

      {/* Input Form */}
      <Card className="p-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Topic Input */}
            <div>
              <Label htmlFor="topic">Topic to Analyze</Label>
              <Input
                id="topic"
                type="text"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="e.g., software engineering, nursing, leadership"
                className="mt-1"
                required
              />
              <p className="text-sm text-gray-500 mt-1">
                Enter any topic you want to analyze for bias
              </p>
            </div>

            {/* Bias Category */}
            <div>
              <Label htmlFor="biasCategory">Bias Category</Label>
              <select
                id="biasCategory"
                value={biasCategory}
                onChange={(e) => setBiasCategory(e.target.value)}
                className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              >
                <option value="">Select bias category...</option>
                <option value="gender">Gender</option>
                <option value="nationality">Nationality</option>
                <option value="race">Race</option>
                <option value="religion">Religion</option>
                <option value="age">Age</option>
                <option value="profession">Profession</option>
              </select>
              <p className="text-sm text-gray-500 mt-1">
                Choose the type of bias to analyze
              </p>
            </div>
          </div>

          {/* Model Selection */}
          <div>
            <Label>AI Models to Compare</Label>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {availableModels.map((model) => (
                <div key={model.id} className="flex items-center space-x-2">
                  <Checkbox
                    id={model.id}
                    checked={selectedModels.includes(model.id)}
                    onCheckedChange={() => handleModelToggle(model.id)}
                  />
                  <Label htmlFor={model.id} className="text-sm">
                    {model.name}
                  </Label>
                </div>
              ))}
            </div>
            <p className="text-sm text-gray-500 mt-2">
              Select one or more models to test and compare
            </p>
          </div>

          {/* Baseline Option */}
          <div className="flex items-center space-x-2">
            <Checkbox
              id="includeBaseline"
              checked={includeBaseline}
              onCheckedChange={(checked) =>
                setIncludeBaseline(checked as boolean)
              }
            />
            <Label htmlFor="includeBaseline">
              Include baseline analysis (Wikipedia content)
            </Label>
            <p className="text-sm text-gray-500">
              Compare model responses against neutral Wikipedia content
            </p>
          </div>

          {/* Submit Button */}
          <div className="flex gap-4">
            <Button
              type="submit"
              disabled={analysisMutation.isPending}
              className="px-6"
            >
              {analysisMutation.isPending
                ? "Testing Models..."
                : `Test ${selectedModels.length} Model${
                    selectedModels.length > 1 ? "s" : ""
                  }`}
            </Button>
            {result && (
              <Button type="button" variant="outline" onClick={resetForm}>
                New Analysis
              </Button>
            )}
          </div>
        </form>

        {/* Example Topics */}
        <div className="mt-6 pt-6 border-t border-gray-200">
          <h3 className="text-sm font-medium text-gray-700 mb-3">
            Quick Examples:
          </h3>
          <div className="flex flex-wrap gap-2">
            {exampleTopics.map((example, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleExampleClick(example)}
                className="text-xs"
              >
                {example.topic} + {example.category}
              </Button>
            ))}
          </div>
        </div>
      </Card>

      {/* Loading State */}
      {analysisMutation.isPending && (
        <Card className="p-6">
          <div className="flex items-center justify-center space-x-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <div className="text-lg">
              <p className="font-medium">
                Testing {selectedModels.length} AI Models...
              </p>
              <p className="text-sm text-gray-600">
                Generating prompts and analyzing responses for comparison
              </p>
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-500 space-y-1">
            <p>🤖 Testing: {selectedModels.join(", ")}</p>
            <p>💬 Collecting model responses...</p>
            <p>📊 Analyzing bias patterns...</p>
            <p>⚖️ Generating comparative analysis...</p>
          </div>
        </Card>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Comparative Summary */}
          <Card className="p-6">
            <h2 className="text-xl font-bold mb-4">
              Comparative Analysis Results
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <h3 className="font-medium text-red-800">Most Biased</h3>
                <p className="text-lg font-bold text-red-600">
                  {result.comparative_analysis.most_biased_model}
                </p>
              </div>
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-medium text-green-800">Least Biased</h3>
                <p className="text-lg font-bold text-green-600">
                  {result.comparative_analysis.least_biased_model}
                </p>
              </div>
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h3 className="font-medium text-blue-800">Models Tested</h3>
                <p className="text-lg font-bold text-blue-600">
                  {result.model_results.length}
                </p>
              </div>
            </div>

            <p className="text-sm text-gray-600">
              <strong>Topic:</strong> {result.topic} |{" "}
              <strong>Category:</strong> {result.bias_category}
            </p>
          </Card>

          {/* Individual Model Results */}
          <div className="space-y-4">
            <h3 className="text-lg font-bold">Individual Model Results</h3>

            {result.model_results.map((modelResult, index) => (
              <Card key={index} className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <h4 className="text-lg font-medium">
                    {modelResult.model_name}
                  </h4>
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      modelResult.summary.bias_detected
                        ? "bg-red-100 text-red-800"
                        : "bg-green-100 text-green-800"
                    }`}
                  >
                    {modelResult.summary.bias_detected
                      ? "Bias Detected"
                      : "No Bias"}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-medium mb-2">Key Metrics</h5>
                    <div className="space-y-2 text-sm">
                      <p>
                        <strong>Confidence:</strong>{" "}
                        {(modelResult.summary.confidence_score * 100).toFixed(
                          1
                        )}
                        %
                      </p>
                      <p>
                        <strong>Avg Sentiment:</strong>{" "}
                        {modelResult.sentiment_analysis.average_sentiment.toFixed(
                          3
                        )}
                      </p>
                      <p>
                        <strong>Biased Keywords:</strong>{" "}
                        {
                          modelResult.bias_indicators.filter(
                            (bi) => bi.bias_detected
                          ).length
                        }
                        /{modelResult.bias_indicators.length}
                      </p>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-medium mb-2">Sentiment Distribution</h5>
                    <div className="grid grid-cols-3 gap-2 text-xs text-center">
                      <div className="bg-red-50 p-2 rounded">
                        <div className="font-bold text-red-600">
                          {(
                            modelResult.sentiment_analysis
                              .sentiment_distribution.negative * 100
                          ).toFixed(1)}
                          %
                        </div>
                        <div>Negative</div>
                      </div>
                      <div className="bg-gray-50 p-2 rounded">
                        <div className="font-bold text-gray-600">
                          {(
                            modelResult.sentiment_analysis
                              .sentiment_distribution.neutral * 100
                          ).toFixed(1)}
                          %
                        </div>
                        <div>Neutral</div>
                      </div>
                      <div className="bg-green-50 p-2 rounded">
                        <div className="font-bold text-green-600">
                          {(
                            modelResult.sentiment_analysis
                              .sentiment_distribution.positive * 100
                          ).toFixed(1)}
                          %
                        </div>
                        <div>Positive</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <p className="text-sm text-gray-600">
                    <strong>Recommendation:</strong>{" "}
                    {modelResult.summary.recommendation}
                  </p>
                </div>
              </Card>
            ))}
          </div>

          {/* Baseline Comparison */}
          {result.baseline_result && (
            <Card className="p-6">
              <h3 className="text-lg font-bold mb-4">
                Baseline Comparison (Wikipedia)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium mb-2">Baseline Metrics</h5>
                  <div className="space-y-2 text-sm">
                    <p>
                      <strong>Confidence:</strong>{" "}
                      {(
                        result.baseline_result.summary.confidence_score * 100
                      ).toFixed(1)}
                      %
                    </p>
                    <p>
                      <strong>Avg Sentiment:</strong>{" "}
                      {result.baseline_result.sentiment_analysis.average_sentiment.toFixed(
                        3
                      )}
                    </p>
                    <p>
                      <strong>Status:</strong>
                      <span
                        className={`ml-2 ${
                          result.baseline_result.summary.bias_detected
                            ? "text-red-600"
                            : "text-green-600"
                        }`}
                      >
                        {result.baseline_result.summary.bias_detected
                          ? "Biased"
                          : "Neutral"}
                      </span>
                    </p>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium mb-2">Model vs Baseline</h5>
                  <div className="space-y-2 text-sm">
                    {Object.entries(
                      result.comparative_analysis.bias_score_differences
                    ).map(([model, diff]) => (
                      <div key={model} className="flex justify-between">
                        <span>{model}:</span>
                        <span
                          className={`font-medium ${
                            diff > 0 ? "text-red-600" : "text-green-600"
                          }`}
                        >
                          {diff > 0 ? "+" : ""}
                          {diff.toFixed(3)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          )}

          <div className="text-sm text-gray-500 text-center">
            Analysis completed in {result.processing_time.toFixed(2)} seconds
          </div>
        </div>
      )}

      {/* Error State */}
      {analysisMutation.isError && (
        <Card className="p-6 bg-red-50 border border-red-200">
          <h3 className="text-red-800 font-medium">Analysis Failed</h3>
          <p className="text-red-600">
            {(analysisMutation.error as Error)?.message ||
              "Unknown error occurred"}
          </p>
        </Card>
      )}
    </div>
  );
};

export default QuickAnalysisComparison;
