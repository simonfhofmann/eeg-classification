# ----------------------------------------------------------------------
# gui_utils.py
#
# Utility functions for displaying GUI elements using Psychopy.
# ----------------------------------------------------------------------

from psychopy import visual, core, event, sound

def setup_window(config):
    """
    Creates and returns the main Psychopy window.
    """
    return visual.Window(
        size=config.SCREEN_SIZE,
        fullscr=config.SCREEN_FULLSCREEN,
        color=config.SCREEN_COLOR,
        units="norm"  # Use 'norm' for easy positioning (-1 to +1)
    )

def show_message(window, text, wait_for_key=True):
    """
    Displays a text message to the participant.
    
    Args:
        window (visual.Window): The Psychopy window.
        text (str): The text to display.
        wait_for_key (bool): If True, waits for any keypress to continue.
    """
    message_stim = visual.TextStim(window, text=text, height=0.1)
    message_stim.draw()
    window.flip()
    
    if wait_for_key:
        event.waitKeys()
    else:
        core.wait(1.0) 

def show_fixation(window, duration):
    """
    Displays a fixation cross '+' for a fixed duration.
    This function actively draws on every frame to allow
    for checking 'escape' keys.
    """
    fixation = visual.TextStim(window, text="+", height=0.2)
    
    timer = core.Clock()
    timer.reset()
    
    while timer.getTime() < duration:
        fixation.draw()
        window.flip()
        
        # Allow user to quit the experiment at any time
        if event.getKeys(keyList=['escape']):
            print("User pressed 'escape'. Quitting.")
            window.close()
            core.quit()

def play_stimulus(filepath):
    """
    Loads and plays a sound file.
    
    Args:
        filepath (str): The full path to the .wav file.
        
    Returns:
        psychopy.sound.Sound: The sound object, so it can be stopped later.
    """
    try:
        stim_sound = sound.Sound(filepath)
        stim_sound.play()
        return stim_sound
    except Exception as e:
        print(f"Error: Could not load or play sound from {filepath}")
        print(f"Details: {e}")
        return None

# In gui_utils.py

def get_likert_responses(window, config, marker_handler):
    """
    Displays two sequential Likert scale questions (Familiarity, then Liking)
    and collects a 1-5 keypress for each within a fixed time window.
    
    Returns:
        (int, int): A tuple of (familiarity_rating, liking_rating).
                    Returns (0) for any question the user ran out of time on.
    """
    
    # --- 1. Setup Stimuli (Text and Scales) ---
    fam_question = visual.TextStim(window, 
                                   text=config.FAMILIARITY_QUESTION,
                                   pos=(0, 0.3), height=0.12)
    lik_question = visual.TextStim(window, 
                                   text=config.LIKING_QUESTION,
                                   pos=(0, 0.3), height=0.12)
    
    # Create scale labels (we'll reuse this for both questions)
    scale_labels = []
    positions = [-0.6, -0.3, 0, 0.3, 0.6] # X-positions for 1, 2, 3, 4, 5
    for i, label in enumerate(config.LIKERT_LABELS):
        scale_labels.append(visual.TextStim(window, text=label,
                             pos=(positions[i], 0.0), height=0.08))

    # Visual feedback for selected answer
    feedback_stim = visual.TextStim(window, text="", color='cyan', 
                                    pos=(0, -0.3), height=0.15)

    # --- 2. Initialize Variables ---
    timer = core.Clock()
    familiarity_rating = 0
    liking_rating = 0
    
    # Clear any keys pressed before the window appeared
    event.clearEvents() 

    # --- 3. Run Familiarity Question Loop ---
    timer.reset()
    while timer.getTime() < config.FAMILIARITY_RESPONSE_DURATION:
        # Draw question and scale
        fam_question.draw()
        for label in scale_labels:
            label.draw()
        window.flip()
        
        # Check for keypress
        keys = event.getKeys(keyList=['1', '2', '3', '4', '5', 'escape'])
        
        if 'escape' in keys:
            window.close()
            core.quit()
            
        if keys:
            familiarity_rating = int(keys[0])
            
            # Show feedback and wait for 0.5s
            feedback_stim.setText(str(familiarity_rating))
            fam_question.draw()
            for label in scale_labels:
                label.draw()
            feedback_stim.draw()
            window.flip()
            core.wait(0.5) # Show feedback
            break # Exit familiarity loop
            
    # If loop finished without a response, rating remains 0 (timeout)

    # --- 4. Run Liking Question Loop ---
    event.clearEvents() # Clear keys from first question
    timer.reset()
    while timer.getTime() < config.LIKING_RESPONSE_DURATION:
        # Draw question and scale
        lik_question.draw()
        for label in scale_labels:
            label.draw()
        window.flip()
        
        # Check for keypress
        keys = event.getKeys(keyList=['1', '2', '3', '4', '5', 'escape'])
        
        if 'escape' in keys:
            window.close()
            core.quit()
            
        if keys:
            liking_rating = int(keys[0])
            
            # Show feedback and wait for 0.5s
            feedback_stim.setText(str(liking_rating))
            lik_question.draw()
            for label in scale_labels:
                label.draw()
            feedback_stim.draw()
            window.flip()
            core.wait(0.5) # Show feedback
            break # Exit liking loop

    # --- 5. Send Markers and Return ---
    # We still send *one* marker that encodes both responses
    # If a question timed out, its value is 0
    
    response_code = (config.MARKERS["RESPONSE_ID_BASE"] + 
                    (familiarity_rating * 10) + 
                    liking_rating)
    marker_handler.send_marker(response_code)
    
    # Send the generic "response made" marker
    marker_handler.send_marker(config.MARKERS["RESPONSE_MADE"])

    return (familiarity_rating, liking_rating)
