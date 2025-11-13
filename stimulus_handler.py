# ----------------------------------------------------------------------
# stimulus_handler.py
#
# Handles finding, loading, and randomizing the stimulus list.
# Manages the persistent "stimulus order" file to ensure
# a participant always gets the same random order, even if
# the experiment is restarted.
#
# *Updated for Two-Pool Paradigm*
# ----------------------------------------------------------------------

import os
import csv
import random

def get_stim_order_filepath(participant_id, log_dir):
    """Generates the full path for a participant's stimulus order file."""
    filename = f"{participant_id}_stimulus_order.csv"
    return os.path.join(log_dir, filename)

def _find_stimuli(stim_root_path, included_genres):
    """
    Internal function to scan directories for .wav files based on genre list.
    """
    all_wav_files = []
    
    if not included_genres:
        print("Warning: No genres provided for a pool. Pool will be empty.")
        return []

    for genre_dir in included_genres:
        full_dir_path = os.path.join(stim_root_path, genre_dir)
        if not os.path.isdir(full_dir_path):
            print(f"Warning: Genre folder '{genre_dir}' not found. Skipping.")
            continue
            
        for filename in os.listdir(full_dir_path):
            if filename.lower().endswith('.wav'):
                all_wav_files.append(os.path.join(full_dir_path, filename))

    if not all_wav_files:
        # This is now just a warning, as one pool might be empty by design
        print(f"Warning: No .wav files found for genres: {included_genres}")
        
    return all_wav_files

def _robust_sample(pool, k, pool_name=""):
    """
    Internal helper to sample k items from a pool.
    Uses repetition if the pool is smaller than k.
    """
    if not pool:
        # If the pool is empty, we can't sample anything
        raise ValueError(f"Cannot sample {k} items: The {pool_name} pool is empty.")

    if len(pool) < k:
        print(f"Warning: Not enough unique stimuli in {pool_name} pool ({len(pool)}). "
              f"Repeating stimuli to fill {k} trials.")
        # Random sampling *with* replacement
        return random.choices(pool, k=k)
    else:
        # Random sampling *without* replacement
        return random.sample(pool, k)

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
            next(reader)  # Skip header
            for row in reader:
                trial_list.append(row[0])
        return trial_list
    except Exception as e:
        print(f"CRITICAL ERROR: Could not *load* stimulus order file from {filepath}")
        print(f"Details: {e}")
        raise

def get_trial_list(participant_id, log_dir, stim_root_path, 
                   familiar_genres, unfamiliar_genres, n_trials):
    """
    This is the main function for this module.
    It builds a trial list by sampling from a familiar and unfamiliar pool.
    - If a list *already exists* for this participant, it loads that list.
    - If NO: It creates a new list and saves it.
    """
    order_filepath = get_stim_order_filepath(participant_id, log_dir)
    
    if os.path.isfile(order_filepath):
        print(f"Found existing stimulus order file. Loading: {order_filepath}")
        trial_list = _load_stimulus_list(order_filepath)
    else:
        print("No stimulus order file found. Creating a new one...")
        
        # 1. Determine trial split
        n_familiar = n_trials // 2
        n_unfamiliar = n_trials - n_familiar
        
        # 2. Find all available .wav files for each pool
        familiar_pool = _find_stimuli(stim_root_path, familiar_genres)
        unfamiliar_pool = _find_stimuli(stim_root_path, unfamiliar_genres)
        
        # 3. Sample from each pool robustly
        familiar_list = _robust_sample(familiar_pool, n_familiar, "FAMILIAR")
        unfamiliar_list = _robust_sample(unfamiliar_pool, n_unfamiliar, "UNFAMILIAR")
        
        # 4. Combine and shuffle the final list
        trial_list = familiar_list + unfamiliar_list
        random.shuffle(trial_list)
        
        # 5. Save this "master plan" to disk immediately
        _save_stimulus_list(order_filepath, trial_list)
        print(f"New stimulus order saved to: {order_filepath}")
        
    return trial_list