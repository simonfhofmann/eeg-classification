# ----------------------------------------------------------------------
# config.py
#
# Configuration file for EEG Experiment.
# ----------------------------------------------------------------------

import os

# --------------------------
# 1. FILE & DIRECTORY PATHS
# --------------------------
# Get the directory where this config file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the root folder containing stimulus subfolders (e.g., /genre_pop, /genre_classical)
STIMULI_ROOT_PATH = os.path.join(BASE_DIR, "stimuli")

# Path to the folder where log files (participant responses) will be saved
LOGS_PATH = os.path.join(BASE_DIR, "logs")

# --------------------------
# 2. HARDWARE SETTINGS (PARALLEL PORT)
# --------------------------
# Memory address of parallel port
PARALLEL_PORT_ADDRESS = 0xDC00

# --------------------------
# 3. EXPERIMENT TIMINGS (in Seconds)
# --------------------------
# Jittered baseline used, as seen in your example script.
# The final duration will be: MEAN + random value between [-JITTER, +JITTER]

BASELINE_DURATION_MEAN = 3.75  # Mean duration of the baseline
BASELINE_DURATION_JITTER = 0.75  # Max +/- jitter (e.g., 3.75 +/- 0.75 -> 3.0s to 4.5s)

# Duration to play each music stimulus
STIMULUS_DURATION = 5.0

# Fixed duration for the participant to enter their ratings
FAMILIARITY_RESPONSE_DURATION = 5.0  # Time for familiarity question
LIKING_RESPONSE_DURATION = 5.0       # Time for liking question

# Duration of the blank screen (Inter-Trial Interval) between trials
ITI_DURATION = 1.5

# --------------------------
# 4. EXPERIMENT PARAMETERS
# --------------------------

# Number of trials to run. This will be the number of songs
# randomly selected from the allowed genres.
N_TRIALS = 2

# Screen settings for Psychopy
SCREEN_SIZE = [1000, 600]
SCREEN_FULLSCREEN = False
SCREEN_COLOR = (-1, -1, -1)  # Black 

# --------------------------
# 5. EVENT MARKER DEFINITIONS
# --------------------------
MARKERS = {
    # --- Experiment Control ---
    "EXPERIMENT_START": 254,
    "EXPERIMENT_END": 255,

    # --- Trial Phases (Static) ---
    "BASELINE_START": 1,
    "STIMULUS_END": 3,
    "RESPONSE_WINDOW_START": 4,
    "RESPONSE_MADE": 5,
    "TRIAL_END_ITI_START": 6,

    # --- Dynamic Markers (Base Numbers) ---
    # Adding trial no to base values
    # e.g., Trial 1 Stimulus Start = 100 + 1 = 101
    #      Trial 40 Stimulus Start = 100 + 40 = 140
    "STIMULUS_START_BASE": 100,

    # Encoding the response as: 200 + (familiarity * 10) + liking
    # e.g., Familiarity=4, Liking=2 -> 200 + 40 + 2 = 242
    # e.g., Familiarity=1, Liking=5 -> 200 + 10 + 5 = 215
    "RESPONSE_ID_BASE": 200,
}

# --------------------------
# 6. GUI TEXT & VISUALS
# --------------------------
# --- Response Screen Prompts ---
FAMILIARITY_QUESTION = "How familiar is this song?"
LIKING_QUESTION = "How much did you like this song?"

# Labels for the 1-5 scales
LIKERT_LABELS = ["1\nNot at all", "2", "3\nNeutral", "4", "5\nVery much"]

# --- Static Screen Text ---
INSTRUCTIONS_TEXT = (
    "Welcome to the experiment.\n\n"
    "You will hear a series of short music clips.\n"
    "Please listen carefully. Keep your eyes focused on the '+' in the center of the screen.\n\n"
    "After each clip, you will be asked to rate:\n"
    "1. How FAMILIAR the song is to you.\n"
    "2. How much you LIKED the song.\n\n"
    "Please use the number keys 1-5 to respond.\n"
    "Press any key to begin."
)

BREAK_TEXT = (
    "You have completed a block.\n\n"
    "Please take a short break.\n"
    "Press any key to continue when you are ready."
)

END_EXPERIMENT_TEXT = (
    "The experiment is now complete.\n\n"
    "Thank you for your participation!"
)

# --------------------------
# 7. DEBUGGING & DEVELOPMENT
# --------------------------
# Set to True to run the experiment without a parallel port.
# Markers will be printed to the console instead of sent to hardware.
#
# !! SET TO False FOR ACTUAL DATA COLLECTION !!
#
DEBUG_MODE = True