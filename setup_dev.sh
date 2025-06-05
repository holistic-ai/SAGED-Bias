#!/bin/bash

# SAGED Bias Analysis Platform - Development Setup Script

echo "🚀 Setting up SAGED Bias Analysis Platform for development..."
echo "================================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

echo "✅ Python and Node.js found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✅ Python dependencies installed"

# Setup frontend
echo "📦 Setting up frontend..."
cd app/frontend

# Install Node.js dependencies
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

echo "✅ Frontend dependencies installed"

# Go back to root directory
cd ../..

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/db
mkdir -p data/app_data/benchmarks
mkdir -p data/app_data/experiments

echo "✅ Directories created"

# Make scripts executable
chmod +x start_app.py
chmod +x setup_dev.sh

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Start the backend: python start_app.py"
echo "2. In another terminal, start the frontend:"
echo "   cd app/frontend && npm run dev"
echo ""
echo "The application will be available at:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo ""
echo "Happy coding! 🚀" 