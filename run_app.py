#!/usr/bin/env python3
"""
Wrapper script to run the Streamlit app with proper environment settings.
"""
import os
import sys

def main():
    # Set environment variables before any imports
    os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Import streamlit first
    import streamlit.bootstrap
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main app file
    app_path = os.path.join(script_dir, 'app.py')
    
    # Prepare command line arguments for Streamlit
    args = [
        "streamlit", "run", app_path, "--server.port=8501",
        "--server.headless=true", "--server.fileWatcherType=none"
    ]
    
    # Add any additional command line arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Run the Streamlit app
    sys.exit(streamlit.bootstrap.main(args[1:]))

if __name__ == "__main__":
    main()
