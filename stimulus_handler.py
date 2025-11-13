# ----------------------------------------------------------------------
# stimulus_handler.py
#
# Handles finding, loading, and randomizing the stimulus list.
# Manages the persistent "stimulus order" file to ensure
# a participant always gets the same random order, even if
# the experiment is restarted.
# ----------------------------------------------------------------------

import os
import csv
import random

def get_stim_order_filepath(participant_id, log_dir):
    """Generates the full path for a participant's stimulus order file."""
    # Example: ./logs/sub-001_stimulus_order.csv
    filename = f"{participant_id}_stimulus_order.csv"
    return os.path.join(log_dir, filename)

def _find_stimuli(stim_root_path, included_genres, n_trials):
    """
    Internal function to scan directories for .wav files.
    """
    all_wav_files = []
    
    if not included_genres:
        # If no genres are specified, scan all subfolders
        target_dirs = [d for d in os.listdir(stim_root_path) 
                       if os.path.isdir(os.path.join(stim_root_path, d))]
    else:
        # Otherwise, only scan the specified genre folders
        target_dirs = included_genres

    for genre_dir in target_dirs:
        full_dir_path = os.path.join(stim_root_path, genre_dir)
        if not os.path.isdir(full_dir_path):
            print(f"Warning: Genre folder '{genre_dir}' not found. Skipping.")
            continue
            
        for filename in os.listdir(full_dir_path):
            if filename.lower().endswith('.wav'):
                # Store the full relative path to the file
                all_wav_files.append(os.path.join(full_dir_path, filename))

    if not all_wav_files:
        raise FileNotFoundError(f"No .wav files found in {stim_root_path}")
        
    # Check if we have enough unique songs
    if len(all_wav_files) < n_trials:
        raise ValueError(
            f"Error: Not enough stimuli found ({len(all_wav_files)}) "
            f"to meet N_TRIALS ({n_trials})."
        )
        
    return all_wav_files

def _save_stimulus_list(filepath, trial_list):
    """Internal function to save the generated list to a CSV."""
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["stimulus_path"])  # Header
            for item in trial_list:
                writer.writerow([item])
    except IOError as e:
        print(f"CRITICAL ERROR: Could not save stimulus order file to {filepath}")
        print(f"Details: {e}")
        raise

def _load_stimulus_list(filepath):
    """Internal function to load a previously saved list."""
    trial_list = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                trial_list.append(row[0])
        return trial_list
    except Exception as e:
        print(f"CRITICAL ERROR: Could not *load* stimulus order file from {filepath}")
        print(f"Details: {e}")
        raise

def get_trial_list(participant_id, log_dir, stim_root_path, genres, n_trials):
    """
    This is the main function for this module.
    It checks if a stimulus order file *already exists* for this participant.
    - If YES: It loads that list (for resuming).
    - If NO: It creates a new randomized list and saves it (for a new run).
    """
    order_filepath = get_stim_order_filepath(participant_id, log_dir)
    
    if os.path.isfile(order_filepath):
        print(f"Found existing stimulus order file. Loading: {order_filepath}")
        trial_list = _load_stimulus_list(order_filepath)
    else:
        print("No stimulus order file found. Creating a new one...")
        # 1. Find all available .wav files
        stimulus_pool = _find_stimuli(stim_root_path, genres, n_trials)
        
        # 2. Shuffle the pool
        random.shuffle(stimulus_pool)
        
        # 3. Select the number you need for the experiment
        trial_list = stimulus_pool[:n_trials]
        
        # 4. Save stimulus order
        _save_stimulus_list(order_filepath, trial_list)
        print(f"New stimulus order saved to: {order_filepath}")
        
    return trial_list