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

# Get the parent directory (one level up from config.py)
PARENT_DIR = os.path.dirname(BASE_DIR)

# Path to the root folder containing stimulus subfolders (e.g., /genre_pop, /genre_classical)
STIMULI_ROOT_PATH = os.path.join(PARENT_DIR, "stimuli")

# Path to the folder where log files (participant responses) will be saved
LOGS_PATH = os.path.join(PARENT_DIR, "logs")

# --------------------------
# 2. HARDWARE SETTINGS (PARALLEL PORT)
# --------------------------
# Memory address of parallel port (Linux)
PARALLEL_PORT_ADDRESS = '/dev/parport0'

# --------------------------
# 3. EXPERIMENT TIMINGS (in Seconds)
# --------------------------
# The final duration will be: MEAN + random value between [-JITTER, +JITTER]

BASELINE_DURATION_MEAN = 2.0
BASELINE_DURATION_JITTER = 0.75

# Duration to play each music stimulus
STIMULUS_DURATION = 5.0

# Fixed duration for the participant to enter their ratings
FAMILIARITY_RESPONSE_DURATION = 10.0
LIKING_RESPONSE_DURATION = 8.0

# Duration of the blank screen (Inter-Trial Interval) between trials
ITI_DURATION = 2.0

# Duration to display feedback after each response
FEEDBACK_DISPLAY_DURATION = 0.5

# --------------------------
# 4. EXPERIMENT PARAMETERS
# --------------------------

# Number of trials to run
N_TRIALS = 60

# Break interval: offer a break every N minutes of experiment time
BREAK_INTERVAL_MINUTES = 1

# Screen settings for Psychopy
SCREEN_SIZE = [1700, 1000]
SCREEN_FULLSCREEN = False
SCREEN_COLOR = (-1, -1, -1)  # Black 

# Position coordinates for Likert scale labels
LIKERT_SCALE_POSITIONS = [-0.6, -0.3, 0, 0.3, 0.6]

# --------------------------
# 5. EVENT MARKER DEFINITIONS
# --------------------------
# Simplified marker protocol - only essential trial phase markers
MARKERS = {
    # --- Experiment Control ---
    "EXPERIMENT_START": 1,
    "EXPERIMENT_END": 2,
    "BREAK_START": 3,
    "BREAK_END": 4,

    # --- Trial Phases ---
    "BASELINE_START": 10,
    "STIMULUS_START": 20,
    "STIMULUS_END": 30,
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
    "Please use the number keys 1-5 to respond.\n\n"
    "You will have the option to take breaks during the experiment.\n\n"
    "Press any key to begin."
)

BREAK_OFFER_TEXT = (
    "You have been working for about {minutes} minutes.\n\n"
    "Would you like to take a break?\n\n"
    "Press SPACE to take a break\n"
    "Press any other key to continue"
)

BREAK_TEXT = (
    "Break time.\n\n"
    "Take your time to rest.\n\n"
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