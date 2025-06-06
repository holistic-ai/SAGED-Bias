import React from "react";
import { Link, useLocation } from "react-router-dom";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation();

  const navigation = [
    { name: "Dashboard", href: "/", icon: "📊" },
    { name: "Quick Analysis", href: "/quick-analysis", icon: "⚡" },
    { name: "Model Comparison", href: "/model-comparison", icon: "⚖️" },
    { name: "Benchmarks", href: "/benchmarks", icon: "📋" },
    { name: "Experiments", href: "/experiments", icon: "🧪" },
    { name: "Analysis", href: "/analysis", icon: "📈" },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-sm shadow-sm border-b border-gray-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">S</span>
                  </div>
                  <div>
                    <h1 className="text-xl font-bold text-gray-900">SAGED</h1>
                    <span className="text-xs text-gray-500 -mt-1 block">
                      Bias Analysis Platform
                    </span>
                  </div>
                </div>
              </div>
              <div className="hidden sm:ml-8 sm:flex sm:space-x-1">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`${
                      location.pathname === item.href
                        ? "bg-blue-50 border-blue-500 text-blue-700"
                        : "border-transparent text-gray-500 hover:bg-gray-50 hover:text-gray-700"
                    } inline-flex items-center px-3 py-2 border-l-2 text-sm font-medium rounded-r-lg transition-all duration-200`}
                  >
                    <span className="mr-2">{item.icon}</span>
                    {item.name}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="max-w-7xl mx-auto py-8 sm:px-6 lg:px-8">
        <div className="px-4 sm:px-0">{children}</div>
      </main>
    </div>
  );
};

export default Layout;
