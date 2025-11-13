# ----------------------------------------------------------------------
# run_experiment.py
#
# Main script 
#
# ----------------------------------------------------------------------

from psychopy import core, event
import random
import sys

# Import 
import os
import config
import logger
import stimulus_handler
import eeg_handler
import gui_utils

def main():
# --- 1. Get Participant Info ---
    try:
        participant_id = input("Enter Participant ID (e.g., sub-001): ")
        if not participant_id:
            print("Participant ID cannot be empty. Exiting.")
            sys.exit()
            
        # --- NEW SECTION: Get Genres ---
        print("-" * 30)
        print(f"Available genres in '{config.STIMULI_ROOT_PATH}':")
        all_dirs = [d for d in os.listdir(config.STIMULI_ROOT_PATH) 
                    if os.path.isdir(os.path.join(config.STIMULI_ROOT_PATH, d))]
        print(f"  {', '.join(all_dirs)}")
        print("-" * 30)
        
        # 1. Get FAMILIAR genres
        fam_genre_input = input("Enter FAMILIAR genres (comma-separated): ")
        familiar_genres = [g.strip() for g in fam_genre_input.split(',')]
        
        # 2. Get UNFAMILIAR genres
        unfam_genre_input = input("Enter UNFAMILIAR genres (comma-separated): ")
        unfamiliar_genres = [g.strip() for g in unfam_genre_input.split(',')]
        
        print(f"Familiar Pool: {familiar_genres}")
        print(f"Unfamiliar Pool: {unfamiliar_genres}")
        # --- END NEW SECTION ---
            
    except KeyboardInterrupt:
        sys.exit()

    print(f"Initializing experiment for: {participant_id}")

    # --- 2. Setup All Components ---
    try:
# Initialize GUI window
        window = gui_utils.setup_window(config)

        # Initialize Parallel Port connection for markers
        marker = eeg_handler.MarkerHandler(config.PARALLEL_PORT_ADDRESS)

        log_filepath = logger.setup_logfile(participant_id, config.LOGS_PATH)

        # Load or create the master stimulus list
        trial_list = stimulus_handler.get_trial_list(
            participant_id=participant_id,
            log_dir=config.LOGS_PATH,
            stim_root_path=config.STIMULI_ROOT_PATH,
            familiar_genres=familiar_genres,     # <-- PASS FAMILIAR LIST
            unfamiliar_genres=unfamiliar_genres, # <-- PASS UNFAMILIAR LIST
            n_trials=config.N_TRIALS
        )

        # Check how many trials are *already* completed
        completed_trials = logger.get_completed_trial_count(participant_id, config.LOGS_PATH)

    except Exception as e:
        print(f"FATAL ERROR during setup: {e}")
        print("Experiment cannot continue. Exiting.")
        sys.exit()

    # --- 3. Show Instructions ---
    gui_utils.show_message(window, config.INSTRUCTIONS_TEXT, wait_for_key=True)

    # --- 4. Run the Experiment ---
    marker.send_marker(config.MARKERS["EXPERIMENT_START"])
    
    if completed_trials > 0:
        print(f"Resuming experiment. {completed_trials} trials already completed.")
        gui_utils.show_message(
            window,
            f"Resuming experiment from trial {completed_trials + 1}.\n\nPress any key to begin.",
            wait_for_key=True
        )
    
    # Start the main loop *from the last completed trial*
    start_index = completed_trials
    
    try:
        for trial_index in range(start_index, len(trial_list)):
            current_trial_num = trial_index + 1
            stimulus_path, origin_pool = trial_list[trial_index]
            
            print(f"--- Starting Trial {current_trial_num} of {len(trial_list)} ---")
            print(f"Stimulus: {stimulus_path} (From: {origin_pool})") 

            # 1. BASELINE
            # Calculate jittered duration
            jitter = random.uniform(-config.BASELINE_DURATION_JITTER, 
                                     config.BASELINE_DURATION_JITTER)
            baseline_duration = config.BASELINE_DURATION_MEAN + jitter
            
            marker.send_marker(config.MARKERS["BASELINE_START"])
            gui_utils.show_fixation(window, baseline_duration)

            # 2. STIMULUS
            # Send the stimulus marker
            stim_marker = config.MARKERS["STIMULUS_START_BASE"] + current_trial_num
            marker.send_marker(stim_marker)
            
            stim_sound = gui_utils.play_stimulus(stimulus_path)
            
            # Use a non-blocking wait to check for 'escape'
            stim_timer = core.Clock()
            while stim_timer.getTime() < config.STIMULUS_DURATION:
                if event.getKeys(keyList=['escape']):
                    raise KeyboardInterrupt("User quit during stimulus")
            
            if stim_sound:
                stim_sound.stop()
            marker.send_marker(config.MARKERS["STIMULUS_END"])

            # 3. RESPONSE
            marker.send_marker(config.MARKERS["RESPONSE_WINDOW_START"])
            
            responses = gui_utils.get_likert_responses(window, config, marker)
            familiarity, liking = responses # (e.g., (4, 5) or (0, 0) if timed out)

            # 4. LOG DATA
            trial_data = [participant_id, current_trial_num, 
                          os.path.basename(stimulus_path), 
                          origin_pool,  
                          familiarity, liking]
            logger.log_trial(log_filepath, trial_data)
            print(f"Logged responses: Familiarity={familiarity}, Liking={liking}")

            marker.send_marker(config.MARKERS["TRIAL_END_ITI_START"])
            gui_utils.show_message(window, text="", wait_for_key=False) 
            core.wait(config.ITI_DURATION) 

    except KeyboardInterrupt:
        print("Experiment interrupted by user. Data saved. Exiting.")
        if 'stim_sound' in locals() and stim_sound:
            stim_sound.stop()
    except Exception as e:
        print(f"A critical error occurred during the trial loop: {e}")
        if 'stim_sound' in locals() and stim_sound:
            stim_sound.stop()
    finally:
        # --- 5. End of Experiment ---
        marker.send_marker(config.MARKERS["EXPERIMENT_END"])
        print("Experiment finished.")
        gui_utils.show_message(window, config.END_EXPERIMENT_TEXT, wait_for_key=True)
        
        # Clean up
        window.close()
        core.quit()

if __name__ == "__main__":
    main()