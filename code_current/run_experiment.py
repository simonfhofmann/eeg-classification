# ----------------------------------------------------------------------
# run_experiment.py
#
# Main script with input validation and break system
#
# ----------------------------------------------------------------------

from psychopy import core, event
import random
import sys
import re

# Import modules
import os
import config
import logger
import stimulus_handler
import eeg_handler
import gui_utils


def validate_participant_id(participant_id):
    """
    Validates that participant ID contains only safe filename characters.
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not re.match(r'^[a-zA-Z0-9_-]+$', participant_id):
        return False
    return True


def validate_genres(genre_list, available_dirs):
    """
    Validates that requested genres exist in the stimuli folder.
    
    Returns:
        (bool, list): (is_valid, list_of_missing_genres)
    """
    missing = []
    for genre in genre_list:
        if genre not in available_dirs:
            missing.append(genre)
    
    return (len(missing) == 0, missing)


def check_genre_overlap(familiar_genres, unfamiliar_genres):
    """
    Checks if there's any overlap between familiar and unfamiliar genre lists.
    
    Returns:
        (bool, list): (has_overlap, list_of_overlapping_genres)
    """
    overlap = set(familiar_genres) & set(unfamiliar_genres)
    return (len(overlap) > 0, list(overlap))


def main():
    # --- 1. Get and Validate Participant Info ---
    try:
        participant_id = input("Enter Participant ID (e.g., sub-001): ").strip()
        
        # Validate participant ID
        if not participant_id:
            print("ERROR: Participant ID cannot be empty. Exiting.")
            sys.exit(1)
            
        if not validate_participant_id(participant_id):
            print("ERROR: Participant ID contains invalid characters.")
            print("Only letters, numbers, hyphens, and underscores are allowed.")
            sys.exit(1)
            
        # --- Get Available Genres ---
        if not os.path.isdir(config.STIMULI_ROOT_PATH):
            print(f"ERROR: Stimuli folder not found at: {config.STIMULI_ROOT_PATH}")
            print("Please create the folder and add genre subfolders with .wav files.")
            sys.exit(1)
            
        print("-" * 50)
        print(f"Available genres in '{config.STIMULI_ROOT_PATH}':")
        all_dirs = [d for d in os.listdir(config.STIMULI_ROOT_PATH) 
                    if os.path.isdir(os.path.join(config.STIMULI_ROOT_PATH, d))]
        
        if not all_dirs:
            print("ERROR: No genre folders found in stimuli directory.")
            print(f"Please add genre folders to: {config.STIMULI_ROOT_PATH}")
            sys.exit(1)
            
        print(f"  {', '.join(all_dirs)}")
        print("-" * 50)
        
        # 1. Get FAMILIAR genres
        fam_genre_input = input("Enter FAMILIAR genres (comma-separated): ").strip()
        if not fam_genre_input:
            print("ERROR: You must specify at least one familiar genre.")
            sys.exit(1)
        familiar_genres = [g.strip() for g in fam_genre_input.split(',') if g.strip()]
        
        # 2. Get UNFAMILIAR genres
        unfam_genre_input = input("Enter UNFAMILIAR genres (comma-separated): ").strip()
        if not unfam_genre_input:
            print("ERROR: You must specify at least one unfamiliar genre.")
            sys.exit(1)
        unfamiliar_genres = [g.strip() for g in unfam_genre_input.split(',') if g.strip()]
        
        # Validate familiar genres
        is_valid, missing = validate_genres(familiar_genres, all_dirs)
        if not is_valid:
            print(f"ERROR: The following FAMILIAR genres were not found: {missing}")
            print(f"Available genres: {all_dirs}")
            sys.exit(1)
            
        # Validate unfamiliar genres
        is_valid, missing = validate_genres(unfamiliar_genres, all_dirs)
        if not is_valid:
            print(f"ERROR: The following UNFAMILIAR genres were not found: {missing}")
            print(f"Available genres: {all_dirs}")
            sys.exit(1)
            
        # Check for overlap
        has_overlap, overlap_list = check_genre_overlap(familiar_genres, unfamiliar_genres)
        if has_overlap:
            print(f"WARNING: The following genres appear in BOTH lists: {overlap_list}")
            confirm = input("Continue anyway? (yes/no): ").strip().lower()
            if confirm not in ['yes', 'y']:
                print("Exiting.")
                sys.exit(0)
        
        print(f"\nParticipant: {participant_id}")
        print(f"Familiar Pool: {familiar_genres}")
        print(f"Unfamiliar Pool: {unfamiliar_genres}")
        print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        sys.exit(0)

    print(f"Initializing experiment for: {participant_id}")

    # --- 2. Setup All Components ---
    marker = None
    window = None
    
    try:
        # Initialize GUI window
        window = gui_utils.setup_window(config)

        # Initialize Parallel Port connection for markers
        marker = eeg_handler.MarkerHandler(config.PARALLEL_PORT_ADDRESS)

        # Setup log file
        log_filepath = logger.setup_logfile(participant_id, config.LOGS_PATH)

        # Load or create the master stimulus list
        trial_list = stimulus_handler.get_trial_list(
            participant_id=participant_id,
            log_dir=config.LOGS_PATH,
            stim_root_path=config.STIMULI_ROOT_PATH,
            familiar_genres=familiar_genres,
            unfamiliar_genres=unfamiliar_genres,
            n_trials=config.N_TRIALS
        )

        # Check how many trials are already completed
        completed_trials = logger.get_completed_trial_count(participant_id, config.LOGS_PATH)

    except Exception as e:
        print(f"FATAL ERROR during setup: {e}")
        print("Experiment cannot continue. Exiting.")
        if window:
            window.close()
        sys.exit(1)

    # --- 3. Show Instructions ---
    gui_utils.show_message(window, config.INSTRUCTIONS_TEXT, wait_for_key=True)

    # --- 4. Run the Experiment ---
    marker.send_marker(config.MARKERS["EXPERIMENT_START"])
    
    # Initialize experiment timer for break tracking
    experiment_timer = core.Clock()
    last_break_time = 0
    break_interval_seconds = config.BREAK_INTERVAL_MINUTES * 60
    
    if completed_trials > 0:
        print(f"Resuming experiment. {completed_trials} trials already completed.")
        gui_utils.show_message(
            window,
            f"Resuming experiment from trial {completed_trials + 1}.\n\nPress any key to begin.",
            wait_for_key=True
        )
    
    # Start the main loop from the last completed trial
    start_index = completed_trials
    stim_sound = None
    
    try:
        for trial_index in range(start_index, len(trial_list)):
            current_trial_num = trial_index + 1
            stimulus_path, origin_pool = trial_list[trial_index]
            
            # --- Check if it's time to offer a break ---
            elapsed_time = experiment_timer.getTime()
            time_since_last_break = elapsed_time - last_break_time
            
            if time_since_last_break >= break_interval_seconds:
                elapsed_minutes = int(elapsed_time / 60)
                print(f"\n--- Offering break after {elapsed_minutes} minutes ---")
                
                took_break = gui_utils.offer_break(window, elapsed_minutes, marker, config)
                if took_break:
                    print("Participant took a break.")
                else:
                    print("Participant chose to continue.")
                
                last_break_time = experiment_timer.getTime()
            
            # --- Run Trial ---
            print(f"\n--- Starting Trial {current_trial_num} of {len(trial_list)} ---")
            print(f"Stimulus: {os.path.basename(stimulus_path)} (From: {origin_pool})") 

            # 1. BASELINE (with jitter)
            jitter = random.uniform(-config.BASELINE_DURATION_JITTER, 
                                     config.BASELINE_DURATION_JITTER)
            baseline_duration = config.BASELINE_DURATION_MEAN + jitter
            
            marker.send_marker(config.MARKERS["BASELINE_START"])
            gui_utils.show_fixation(window, baseline_duration)

            # 2. STIMULUS
            marker.send_marker(config.MARKERS["STIMULUS_START"])
            
            stim_sound = gui_utils.play_stimulus(stimulus_path)
            
            # Use a non-blocking wait to check for 'escape'
            stim_timer = core.Clock()
            while stim_timer.getTime() < config.STIMULUS_DURATION:
                if event.getKeys(keyList=['escape']):
                    raise KeyboardInterrupt("User quit during stimulus")
            
            if stim_sound:
                stim_sound.stop()
            marker.send_marker(config.MARKERS["STIMULUS_END"])

            # 3. RESPONSE (no markers sent here)
            responses = gui_utils.get_likert_responses(window, config)
            familiarity, liking = responses

            # 4. LOG DATA
            trial_data = [
                participant_id, 
                current_trial_num, 
                os.path.basename(stimulus_path), 
                origin_pool,  
                familiarity, 
                liking
            ]
            logger.log_trial(log_filepath, trial_data)
            
            if familiarity == 0 or liking == 0:
                print(f"WARNING: Timeout on response (Familiarity={familiarity}, Liking={liking})")
            else:
                print(f"Logged responses: Familiarity={familiarity}, Liking={liking}")

            # 5. INTER-TRIAL INTERVAL
            gui_utils.show_message(window, text="", wait_for_key=False) 
            core.wait(config.ITI_DURATION) 

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user. Data saved. Exiting.")
        if stim_sound:
            stim_sound.stop()
    except Exception as e:
        print(f"\nCRITICAL ERROR during trial loop: {e}")
        if stim_sound:
            stim_sound.stop()
    finally:
        # --- 5. End of Experiment ---
        marker.send_marker(config.MARKERS["EXPERIMENT_END"])
        
        print("\nExperiment finished.")
        gui_utils.show_message(window, config.END_EXPERIMENT_TEXT, wait_for_key=True)
        
        # Clean up
        window.close()
        core.quit()

if __name__ == "__main__":
    main()