#!/usr/bin/env python3
"""
Simple script to start the OmniParser API server
"""
import sys
import argparse
from omniparser_api import api


def main():
    parser = argparse.ArgumentParser(description="Start OmniParser API Server")
    parser.add_argument("--host", type=str,
                        default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")

    args = parser.parse_args()

    print(f"🚀 Starting OmniParser API server...")
    print(f"📡 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {args.reload}")
    print(f"📖 API docs: http://{args.host}:{args.port}/docs")
    print(f"❤️  Health check: http://{args.host}:{args.port}/health")

    try:
        api.run(host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
