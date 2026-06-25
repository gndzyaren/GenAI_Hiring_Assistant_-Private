"""
Run the FastAPI backend server.
Execute: python run_backend.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", "8000"))
    print(f"\n🚀 Starting Assessment Engine backend on http://{host}:{port}")
    print(f"📄 API docs: http://{host}:{port}/docs\n")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
