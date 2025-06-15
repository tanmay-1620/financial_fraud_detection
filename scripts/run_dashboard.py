import os
import subprocess
import sys

def run_dashboard():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the dashboard script
    dashboard_path = os.path.join(current_dir, "dashboard.py")
    
    # Ensure we're in the correct directory
    os.chdir(current_dir)
    
    # Run the dashboard using streamlit
    try:
        # Use the full path to streamlit
        streamlit_cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
        print(f"Running command: {' '.join(streamlit_cmd)}")
        subprocess.run(streamlit_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_dashboard() 