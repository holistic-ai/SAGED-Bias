import React from "react";

const Dashboard: React.FC = () => {
  return (
    <div className="space-y-8">
      <div className="border-b border-gray-200 pb-6">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Overview of your bias analysis projects and recent activity
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white overflow-hidden shadow-sm rounded-xl border border-gray-100 hover:shadow-md transition-shadow">
          <div className="p-6">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-blue-100 rounded-full">
                <span className="text-2xl">📋</span>
              </div>
              <div>
                <div className="text-2xl font-bold text-blue-600">12</div>
                <p className="text-sm text-gray-500">Total Benchmarks</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow-sm rounded-xl border border-gray-100 hover:shadow-md transition-shadow">
          <div className="p-6">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-green-100 rounded-full">
                <span className="text-2xl">🧪</span>
              </div>
              <div>
                <div className="text-2xl font-bold text-green-600">3</div>
                <p className="text-sm text-gray-500">Active Experiments</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow-sm rounded-xl border border-gray-100 hover:shadow-md transition-shadow">
          <div className="p-6">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-purple-100 rounded-full">
                <span className="text-2xl">📈</span>
              </div>
              <div>
                <div className="text-2xl font-bold text-purple-600">28</div>
                <p className="text-sm text-gray-500">Completed Analyses</p>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow-sm rounded-xl border border-gray-100 hover:shadow-md transition-shadow">
          <div className="p-6">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-red-100 rounded-full">
                <span className="text-2xl">⚠️</span>
              </div>
              <div>
                <div className="text-2xl font-bold text-red-600">5</div>
                <p className="text-sm text-gray-500">Bias Alerts</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white shadow-sm rounded-xl border border-gray-100">
        <div className="p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">
            Recent Activity
          </h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-green-50 rounded-lg border border-green-100">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-green-100 rounded-full">
                  <span className="text-green-600">✅</span>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-900">
                    Experiment "Gender Bias in Job Descriptions" completed
                  </span>
                  <p className="text-xs text-gray-500">
                    Successfully analyzed 1,250 job descriptions
                  </p>
                </div>
              </div>
              <span className="text-xs text-gray-400 font-medium">
                2 hours ago
              </span>
            </div>

            <div className="flex items-center justify-between p-4 bg-blue-50 rounded-lg border border-blue-100">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-100 rounded-full">
                  <span className="text-blue-600">🔄</span>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-900">
                    Benchmark "Nationality Stereotypes" is being processed
                  </span>
                  <p className="text-xs text-gray-500">
                    Processing 850 text samples
                  </p>
                </div>
              </div>
              <span className="text-xs text-gray-400 font-medium">
                4 hours ago
              </span>
            </div>

            <div className="flex items-center justify-between p-4 bg-yellow-50 rounded-lg border border-yellow-100">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-yellow-100 rounded-full">
                  <span className="text-yellow-600">⚠️</span>
                </div>
                <div>
                  <span className="text-sm font-medium text-gray-900">
                    High bias detected in "Professional Descriptions" analysis
                  </span>
                  <p className="text-xs text-gray-500">
                    Bias rate: 23.4% above threshold
                  </p>
                </div>
              </div>
              <span className="text-xs text-gray-400 font-medium">
                1 day ago
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
