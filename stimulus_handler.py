# ----------------------------------------------------------------------
# stimulus_handler.py
# Manages the selection and ordering of stimuli
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
        print(f"Warning: No .wav files found for genres: {included_genres}")
        
    return all_wav_files

def _robust_sample(pool, k, pool_name=""):
    """
    Internal helper to sample k items from a pool.
    Uses repetition if the pool is smaller than k.
    """
    if not pool:
        raise ValueError(f"Cannot sample {k} items: The {pool_name} pool is empty.")

    if len(pool) < k:
        print(f"Warning: Not enough unique stimuli in {pool_name} pool ({len(pool)}). "
              f"Repeating stimuli to fill {k} trials.")
        return random.choices(pool, k=k)
    else:
        return random.sample(pool, k)

def _save_stimulus_list(filepath, trial_list_tuples):
    """Internal function to save the generated list (with pool labels) to a CSV."""
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:  # <-- Add encoding='utf-8'
            writer = csv.writer(f)
            writer.writerow(["stimulus_path", "origin_pool"])
            for item_tuple in trial_list_tuples:
                writer.writerow(item_tuple)
    except IOError as e:
        print(f"CRITICAL ERROR: Could not save stimulus order file to {filepath}")
        print(f"Details: {e}")
        raise

def _load_stimulus_list(filepath):
    """Internal function to load a previously saved list (now as tuples)."""
    trial_list_tuples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:  # <-- Add encoding='utf-8'
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row: # Make sure row is not empty
                    trial_list_tuples.append((row[0], row[1])) 
        return trial_list_tuples
    except Exception as e:
        print(f"CRITICAL ERROR: Could not *load* stimulus order file from {filepath}")
        print(f"Details: {e}")
        raise

def get_trial_list(participant_id, log_dir, stim_root_path, 
                   familiar_genres, unfamiliar_genres, n_trials=None):
    """
    This is the main function for this module.
    Builds a trial list by using ALL available items from both pools.
    n_trials parameter is ignored - total trials = total available songs.
    """
    order_filepath = get_stim_order_filepath(participant_id, log_dir)
    
    if os.path.isfile(order_filepath):
        print(f"Found existing stimulus order file. Loading: {order_filepath}")
        trial_list = _load_stimulus_list(order_filepath)
    else:
        print("No stimulus order file found. Creating a new one...")
        
        # 1. Find all available .wav files for each pool
        familiar_pool = _find_stimuli(stim_root_path, familiar_genres)
        unfamiliar_pool = _find_stimuli(stim_root_path, unfamiliar_genres)
        
        # 2. Use ALL songs from each pool (no sampling)
        print(f"Using all {len(familiar_pool)} songs from familiar pool")
        print(f"Using all {len(unfamiliar_pool)} songs from unfamiliar pool")
        
        # 3. Tag items with their origin pool
        familiar_list_tagged = [(path, 'familiar_pool') for path in familiar_pool]
        unfamiliar_list_tagged = [(path, 'unfamiliar_pool') for path in unfamiliar_pool]
        
        # 4. Combine and shuffle the final list of tuples
        trial_list = familiar_list_tagged + unfamiliar_list_tagged
        random.shuffle(trial_list)
        
        print(f"Total trials: {len(trial_list)}")
        
        # 5. Save this "master plan" to disk immediately
        _save_stimulus_list(order_filepath, trial_list)
        print(f"New stimulus order saved to: {order_filepath}")
        
    return trial_list