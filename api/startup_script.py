"""Startup script for installing dependencies and launching the FastAPI application with Gunicorn."""

import subprocess
import sys
import os

def main():
    """Install required packages and start FastAPI server with Gunicorn workers."""
    # Install dependencies
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Start the application
    os.system('gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker main:app')

if __name__ == "__main__":
    main()
