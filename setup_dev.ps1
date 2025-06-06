Write-Host "🚀 Setting up SAGED Bias Analysis Platform for development..."
Write-Host "================================================================"

# Check if Python is installed
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "❌ Python 3.10+ is required but not installed."
    exit 1
}

# Check Python version
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pythonVersion -lt [version]"3.10") {
    Write-Host "❌ Python 3.10+ is required, but Python $pythonVersion is installed."
    Write-Host "Please upgrade Python to version 3.10 or higher."
    exit 1
} else {
    Write-Host "✅ Python $pythonVersion detected (required: 3.10+)"
}

# Check if Node.js is installed
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    Write-Host "❌ Node.js is required but not installed."
    exit 1
}
Write-Host "✅ Python and Node.js found"

# Install Python dependencies
Write-Host "📦 Installing Python dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Python dependencies"
    exit 1
}
Write-Host "✅ Python dependencies installed"

# Setup frontend
Write-Host "📦 Setting up frontend..."
Push-Location app/frontend
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Node.js dependencies"
    Pop-Location
    exit 1
}
Write-Host "✅ Frontend dependencies installed"
Pop-Location

# Create necessary directories
Write-Host "📁 Creating necessary directories..."
New-Item -ItemType Directory -Force -Path "data\db" | Out-Null
New-Item -ItemType Directory -Force -Path "data\app_data\benchmarks" | Out-Null
New-Item -ItemType Directory -Force -Path "data\app_data\experiments" | Out-Null
Write-Host "✅ Directories created"

Write-Host ""
Write-Host "🎉 Setup complete!"
Write-Host ""
Write-Host "To start the application:"
Write-Host "1. Start the backend: python start_full_app.py"
Write-Host "2. In another terminal, start the frontend:"
Write-Host "   cd app/frontend && npm run dev"
Write-Host ""
Write-Host "The application will be available at:"
Write-Host "- Frontend: http://localhost:3000"
Write-Host "- Backend API: http://localhost:8000"
Write-Host "- API Documentation: http://localhost:8000/docs"
Write-Host ""
Write-Host "Happy coding! 🚀"