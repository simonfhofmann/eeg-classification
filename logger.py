# ----------------------------------------------------------------------
# logger.py
#
# Handles behavioral data logging for the experiment.
# ----------------------------------------------------------------------

import os
import csv

def get_log_filepath(participant_id, log_dir):
    """Generates the full path for a participant's data log file."""
    # Example: ./logs/sub-001_data.csv
    filename = f"{participant_id}_data.csv"
    return os.path.join(log_dir, filename)

def setup_logfile(participant_id, log_dir):
    """
    Creates the log directory and file if they don't exist.
    Writes the header row to the CSV file *only if* the file is new.
    """
    log_filepath = get_log_filepath(participant_id, log_dir)
    
    # 1. Create the log directory if it's not already there
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. Check if the file already exists.
    file_exists = os.path.isfile(log_filepath)
    
    # 3. Open in 'append' mode 
    try:
        with open(log_filepath, 'a', newline='') as f:
            if not file_exists:
                writer = csv.writer(f)
                # --- NEW: Added origin_pool column ---
                header = ["participant_id", "trial_num", "stimulus_name", 
                          "origin_pool", "familiarity_rating", "liking_rating"]
                writer.writerow(header)
        return log_filepath
    except IOError as e:
        print(f"Error: Could not create or open log file at {log_filepath}")
        print(f"Details: {e}")
        raise

def log_trial(log_filepath, trial_data):
    """
    Logs a single trial's data. Opens the file in append mode,
    writes one row, and closes it.
    
    Args:
        log_filepath (str): The full path to the log file.
        trial_data (list or tuple): The data to write for the row.
                                    e.g., ["sub-001", 18, "song_X.wav", 4, 5]
    """
    try:
        with open(log_filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(trial_data)
    except IOError as e:
        print(f"Error: Could not write trial data to {log_filepath}")
        print(f"Details: {e}")
        

def get_completed_trial_count(participant_id, log_dir):
    """
    Checks an existing log file to see how many trials are already completed.
    This is the core of the "resume" logic.
    """
    log_filepath = get_log_filepath(participant_id, log_dir)
    
    if not os.path.isfile(log_filepath):
        return 0  
        
    try:
        with open(log_filepath, 'r') as f:
            reader = csv.reader(f)
            row_count = len(list(reader))
            completed_count = max(0, row_count - 1)
            return completed_count
    except Exception as e:
        print(f"Error: Could not read log file {log_filepath} to count trials.")
        print(f"Details: {e}")
        raise