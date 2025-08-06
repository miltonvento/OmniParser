#!/bin/bash
# Simple bash script to start the OmniParser API server

echo "🚀 Starting OmniParser API Server..."

# Default values
HOST="0.0.0.0"
PORT="8000"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--host HOST] [--port PORT]"
            echo "  --host HOST    Host to bind to (default: 0.0.0.0)"
            echo "  --port PORT    Port to bind to (default: 8000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "📡 Host: $HOST"
echo "🔌 Port: $PORT"
echo "📖 API docs will be available at: http://$HOST:$PORT/docs"
echo "❤️  Health check: http://$HOST:$PORT/health"
echo ""

# Start the server
python omniparser_api.py --host "$HOST" --port "$PORT"