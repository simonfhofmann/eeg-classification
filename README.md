# EEG Familiarity Recording Script

This project is a Python-based experimental paradigm for collecting EEG and behavioral data to classify music familiarity. Participants listen to music excerpts and provide familiarity/liking ratings, while event markers are sent to an EEG system via a parallel port.

## Purpose

The goal is to collect synchronized EEG and behavioral data to train a machine learning model capable of distinguishing between "familiar" and "unfamiliar" cognitive states based on brain activity.

**2. Create Conda Environment**
This project uses a Conda environment. The required `environment.yml` file is included in the repository.

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment (you must do this every time)
conda activate eeg_lab
```

**3. CRITICAL HARDWARE DRIVER**
This script requires a 64-bit parallel port driver (`inpoutx64.dll`) to send EEG markers. This driver is **not** installed by Conda.
It is however included in the repository.

The script will fail to connect to the hardware without this file (unless `DEBUG_MODE` is on).

## How to Run

**1. Configure the Experiment**
Before running, check `config.py` to:

  * Set `DEBUG_MODE = True` to test the script without hardware (markers will print to the console).
  * Set `DEBUG_MODE = False` for actual data collection.
  * Verify `PARALLEL_PORT_ADDRESS` matches your hardware (e.g., `0xDC00`).
  * Adjust trial timings, `N_TRIALS`, etc. as needed.

**2. Run the Script**
Ensure the `conda activate eeg_lab` environment is active and the `inpoutx64.dll` file is in place.

```bash
python run_experiment.py
```

**3. Follow Prompts**
The script will ask for:

1.  `Enter Participant ID (e.g., sub-001):`
2.  `Enter FAMILIAR genres (comma-separated):` (e.g., `pop, rock`)
3.  `Enter UNFAMILIAR genres (comma-separated):` (e.g., `baroque, ambient`)

The experiment will then load the stimuli, create the log files, and wait for you to start the EEG recording and for the participant to press a key.

## File Structure

  * **`run_experiment.py`**: The main executable script. Runs the experiment.
  * **`config.py`**: Central control panel. **Edit all parameters here.**
  * **`eeg_handler.py`**: Manages the parallel port connection (real or debug).
  * **`gui_utils.py`**: Manages all Psychopy screen drawing and keyboard input.
  * **`stimulus_handler.py`**: Manages finding, loading, and randomizing the stimulus list.
  * **`logger.py`**: Manages writing behavioral data and resuming after a crash.
  * **`environment.yml`**: Conda environment file for collaborators.
  * **`stimuli/`**: (You create) Root folder containing genre subfolders with `.wav` files.
  * **`logs/`**: (Auto-created) Output folder for all behavioral data (`_data.csv`) and stimulus lists (`_stimulus_order.csv`).

<!-- end list -->

```
```
## TODO
Before playing starts adjust the volume (with sample track)

after rating is given show the rating in an extra screen and ask for confirmation (via enter) if user doesnt confirm start the confirmation 