# EEG Marker Reference Guide

**Generated for**: EEG Music Familiarity Experiment  
**Last Updated**: December 2024

---

## Overview

This document describes all event markers sent to the EEG system during the experiment. Markers are sent via parallel port and synchronized with the continuous EEG recording.

**Important**: Trial-specific information (trial number, stimulus identity, responses) is NOT encoded in markers. Instead, this information is stored in the behavioral log CSV file (`<participant_id>_data.csv`) and stimulus order file (`<participant_id>_stimulus_order.csv`). To synchronize EEG data with trial information, use the marker timestamps along with the trial order from the CSV files.

---

## Marker Protocol Summary

| Marker Value | Marker Name         | Description                                    |
|--------------|---------------------|------------------------------------------------|
| 1            | EXPERIMENT_START    | Sent once at the beginning of the experiment   |
| 2            | EXPERIMENT_END      | Sent once at the end of the experiment         |
| 3            | BREAK_START         | Participant chose to take a break              |
| 4            | BREAK_END           | Participant resumed after break                |
| 10           | BASELINE_START      | Start of baseline/fixation period (jittered)   |
| 20           | STIMULUS_START      | Audio stimulus playback begins                 |
| 30           | STIMULUS_END        | Audio stimulus playback ends                   |

---

## Detailed Marker Descriptions

### Experiment Control Markers

#### EXPERIMENT_START (Marker: 1)
- **When**: Sent immediately after instructions screen, before first trial
- **Purpose**: Marks the beginning of the recording session
- **Notes**: Use this as the reference point for the entire experiment timeline

#### EXPERIMENT_END (Marker: 2)
- **When**: Sent after the last trial is completed
- **Purpose**: Marks the end of the recording session
- **Notes**: Sent regardless of whether experiment completed normally or was interrupted

#### BREAK_START (Marker: 3)
- **When**: Sent when participant accepts a break offer (presses SPACE)
- **Purpose**: Marks the beginning of a participant-initiated break
- **Notes**: Breaks are offered approximately every 10 minutes. Break duration is variable and determined by participant

#### BREAK_END (Marker: 4)
- **When**: Sent when participant resumes after a break (presses any key)
- **Purpose**: Marks the end of a break period and resumption of the experiment
- **Notes**: The next trial will begin immediately after this marker

---

### Trial Phase Markers

Each trial consists of the following phases in order:

#### BASELINE_START (Marker: 10)
- **When**: At the start of each baseline/fixation period
- **Duration**: 8.0 seconds ± 0.75 seconds (jittered)
- **Purpose**: Marks the beginning of the resting state baseline
- **Visual**: Participant sees a fixation cross (+)
- **Notes**: Duration is randomized per trial to prevent anticipation effects

#### STIMULUS_START (Marker: 20)
- **When**: Immediately when audio playback begins
- **Duration**: 32.0 seconds (fixed)
- **Purpose**: Marks the onset of the music stimulus
- **Visual**: Participant continues viewing the fixation cross
- **Notes**: The specific stimulus played is identified by matching this marker's timestamp with the trial number in the CSV log

#### STIMULUS_END (Marker: 30)
- **When**: Immediately when audio playback ends
- **Purpose**: Marks the offset of the music stimulus
- **Notes**: Response collection begins immediately after this marker (no marker sent for response phase)

---

## Trial Structure Timeline

```
Trial N:
  [10] BASELINE_START
       └─> Fixation cross displayed for ~8 seconds (jittered)
  
  [20] STIMULUS_START
       └─> Audio playback for 32 seconds
  
  [30] STIMULUS_END
       └─> Familiarity question (10 seconds)
       └─> Liking question (8 seconds)
       └─> ITI blank screen (2 seconds)
  
Trial N+1:
  [10] BASELINE_START
       └─> ...
```

---

## What is NOT in the Markers

The following information is **not** encoded in markers and must be retrieved from CSV log files:

1. **Trial Number**: Not encoded. Use marker sequence and timestamps to determine trial order.
2. **Stimulus Identity**: Not encoded. Match marker timestamps to trial numbers in `<participant_id>_stimulus_order.csv`.
3. **Stimulus Pool** (familiar/unfamiliar): Stored in `origin_pool` column of the stimulus order CSV.
4. **Response Values**: Not encoded. Familiarity and liking ratings are stored only in the behavioral log CSV.
5. **Response Timing**: Not encoded. Response onset/offset markers are not sent.

---

## Synchronizing EEG with Behavioral Data

To analyze your data:

1. **Extract marker timestamps** from your EEG data file
2. **Count markers** to identify trial numbers:
   - First `BASELINE_START` (10) = Trial 1
   - Second `BASELINE_START` (10) = Trial 2
   - etc.

3. **Match to CSV files**:
   - `<participant_id>_data.csv` contains trial_num, stimulus_name, familiarity, liking
   - `<participant_id>_stimulus_order.csv` contains the full stimulus order with pool labels

4. **Epoch your data**:
   - Baseline period: from `BASELINE_START` (10) to `STIMULUS_START` (20)
   - Stimulus period: from `STIMULUS_START` (20) to `STIMULUS_END` (30)

---

## Break Periods

Participants are offered breaks approximately every 10 minutes. Break periods **are marked** in the EEG data:

- `BREAK_START` (3): Participant pressed SPACE and accepted the break
- `BREAK_END` (4): Participant pressed any key to resume

### Handling Breaks in Analysis

If a participant **declines** a break offer, no markers are sent and the next trial begins immediately.

If a participant **accepts** a break:
```
... [30] STIMULUS_END (end of previous trial)
    [3] BREAK_START
    ... (variable duration - participant rests)
    [4] BREAK_END
    [10] BASELINE_START (next trial begins)
```

**Recommendation for analysis**: 
- Exclude data segments between BREAK_START and BREAK_END from continuous analysis
- Be aware that the first trial after a break may show different baseline characteristics
- Consider marking the first 1-2 trials post-break separately if analyzing sustained attention effects

---

## Configuration

These marker values are defined in `config.py` and can be modified if needed:

```python
MARKERS = {
    "EXPERIMENT_START": 1,
    "EXPERIMENT_END": 2,
    "BREAK_START": 3,
    "BREAK_END": 4,
    "BASELINE_START": 10,
    "STIMULUS_START": 20,
    "STIMULUS_END": 30,
}
```

**Warning**: If you change marker values, update this reference document accordingly.

---

## Troubleshooting

### Missing Markers
- Check that `DEBUG_MODE = False` in `config.py` for actual data collection
- Verify parallel port connection: `ls /dev/parport*`
- Check parallel port permissions: `sudo chmod 666 /dev/parport0`

### Marker Timing Verification
- In debug mode, markers are printed to console with timestamps
- Verify marker order matches expected trial structure
- Check that BASELINE_START always precedes STIMULUS_START

### Data Analysis Tips
- Use `STIMULUS_START` as your primary time-locking event for ERPs
- Baseline correction: use 200ms before `STIMULUS_START` or the full baseline period
- For familiarity classification: segment trials based on the `origin_pool` column from CSV

---

## Example Analysis Workflow

```python
# Pseudo-code for typical analysis

# 1. Load EEG data and extract marker timestamps
eeg_data, markers = load_eeg_data('participant_001.eeg')
baseline_starts = markers[markers.value == 10].timestamp
stimulus_starts = markers[markers.value == 20].timestamp
stimulus_ends = markers[markers.value == 30].timestamp

# 2. Load behavioral data
behavior = pd.read_csv('logs/sub-001_data.csv')
stim_order = pd.read_csv('logs/sub-001_stimulus_order.csv')

# 3. Merge behavioral and neural data
for trial_num, stim_start in enumerate(stimulus_starts, start=1):
    trial_behavior = behavior[behavior.trial_num == trial_num]
    pool_label = trial_behavior.origin_pool.values[0]
    familiarity = trial_behavior.familiarity_rating.values[0]
    
    # Epoch EEG data around stimulus onset
    epoch = extract_epoch(eeg_data, stim_start, tmin=-0.2, tmax=2.0)
    
    # Analyze...
```

---

## Contact & Support

For questions about the marker protocol or data analysis, refer to the experiment documentation or contact the research team.