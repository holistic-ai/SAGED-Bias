import React from "react";

const Experiments: React.FC = () => {
  return (
    <div>
      <div className="mb-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Experiments</h1>
            <p className="mt-2 text-gray-600">
              Run and monitor bias analysis experiments
            </p>
          </div>
          <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium">
            New Experiment
          </button>
        </div>
      </div>

      {/* Experiments List */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="space-y-4">
            {/* Running experiment */}
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-gray-900">
                    Gender Bias Analysis - GPT-4
                  </h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Benchmark: Gender Bias in Job Descriptions
                  </p>
                  <div className="flex items-center mt-2 space-x-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      Running
                    </span>
                    <span className="text-xs text-gray-500">
                      Started 2 hours ago
                    </span>
                  </div>
                  {/* Progress bar */}
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Progress</span>
                      <span className="text-gray-900">65%</span>
                    </div>
                    <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: "65%" }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      Extracting features...
                    </p>
                  </div>
                </div>
                <div className="flex space-x-2 ml-4">
                  <button className="text-red-600 hover:text-red-800 text-sm">
                    Stop
                  </button>
                </div>
              </div>
            </div>

            {/* Completed experiment */}
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">
                    Nationality Bias - Claude 3
                  </h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Benchmark: Nationality Stereotypes
                  </p>
                  <div className="flex items-center mt-2 space-x-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      Completed
                    </span>
                    <span className="text-xs text-gray-500">
                      Completed 1 day ago • Duration: 45 min
                    </span>
                  </div>
                  <div className="mt-2">
                    <p className="text-sm text-gray-600">
                      <span className="font-medium">Results:</span> 3 features
                      with significant bias detected
                    </p>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button className="text-blue-600 hover:text-blue-800 text-sm">
                    View Results
                  </button>
                  <button className="text-gray-600 hover:text-gray-800 text-sm">
                    Export
                  </button>
                  <button className="text-red-600 hover:text-red-800 text-sm">
                    Delete
                  </button>
                </div>
              </div>
            </div>

            {/* Failed experiment */}
            <div className="border border-gray-200 rounded-lg p-4">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">
                    Age Bias Analysis - Llama 2
                  </h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Benchmark: Age-Related Bias in Healthcare
                  </p>
                  <div className="flex items-center mt-2 space-x-4">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                      Failed
                    </span>
                    <span className="text-xs text-gray-500">
                      Failed 3 hours ago
                    </span>
                  </div>
                  <div className="mt-2">
                    <p className="text-sm text-red-600">
                      Error: API rate limit exceeded during generation phase
                    </p>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <button className="text-blue-600 hover:text-blue-800 text-sm">
                    Retry
                  </button>
                  <button className="text-red-600 hover:text-red-800 text-sm">
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Experiments;
