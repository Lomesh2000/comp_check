#!/bin/bash
# Convenience script to run the LexGuard API

set -e

echo "================================"
echo "LexGuard API - Startup Script"
echo "================================"
echo ""

# Check if running in venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Not in a virtual environment"
    echo "   Consider activating: source venv/bin/activate"
    echo ""
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import fastapi" 2>/dev/null || {
    echo "❌ FastAPI not installed"
    echo "   Run: pip install -r requirements.txt"
    exit 1
}

echo "✓ Dependencies OK"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}

echo "Starting API server on http://localhost:$PORT"
echo "Interactive docs: http://localhost:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the API
python -m src.api.app
