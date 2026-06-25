"""
Run the Streamlit frontend.
Execute: python run_frontend.py
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "app.py")
    print("\n🌐 Starting Streamlit frontend...")
    print("Opening at http://localhost:8501\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", frontend_path,
        "--server.port", "8501",
        "--server.headless", "false",
        "--theme.base", "dark",
    ])
