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
        units="norm"
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

def offer_break(window, elapsed_minutes, marker_handler, config):
    """
    Offers the participant a break and waits for their decision.
    Sends break markers if participant accepts.
    
    Args:
        window: The Psychopy window
        elapsed_minutes: How many minutes have elapsed
        marker_handler: The marker handler object
        config: The config module
        
    Returns:
        bool: True if participant took a break, False otherwise
    """
    from psychopy import event
    
    message = visual.TextStim(
        window, 
        text=f"You have been working for about {elapsed_minutes} minutes.\n\n"
             "Would you like to take a break?\n\n"
             "Press SPACE to take a break\n"
             "Press any other key to continue",
        height=0.08
    )
    
    message.draw()
    window.flip()
    
    keys = event.waitKeys()
    
    if 'escape' in keys:
        window.close()
        core.quit()
    
    if 'space' in keys:
        # Participant chose to take a break
        marker_handler.send_marker(config.MARKERS["BREAK_START"])
        
        # Show break message
        break_msg = visual.TextStim(window, text=config.BREAK_TEXT, height=0.08)
        break_msg.draw()
        window.flip()
        event.waitKeys()
        
        marker_handler.send_marker(config.MARKERS["BREAK_END"])
        return True
    
    return False

def get_likert_responses(window, config):
    """
    Displays two sequential Likert scale questions (Familiarity, then Liking)
    and collects a 1-5 keypress for each within a fixed time window.
    No markers are sent - responses are only logged to CSV.
    
    Returns:
        (int, int): A tuple of (familiarity_rating, liking_rating).
                    Returns (0, 0) if BOTH questions time out.
    """
    
    # --- 1. Setup Stimuli ---
    fam_question = visual.TextStim(
        window, 
        text=config.FAMILIARITY_QUESTION,
        pos=(0, 0.3), 
        height=0.12
    )
    lik_question = visual.TextStim(
        window, 
        text=config.LIKING_QUESTION,
        pos=(0, 0.3), 
        height=0.12
    )
    
    # Create scale labels
    scale_labels = []
    for i, label in enumerate(config.LIKERT_LABELS):
        scale_labels.append(visual.TextStim(
            window, 
            text=label,
            pos=(config.LIKERT_SCALE_POSITIONS[i], 0.0), 
            height=0.08
        ))

    # Visual feedback for selected answer
    feedback_stim = visual.TextStim(
        window, 
        text="", 
        color='cyan', 
        pos=(0, -0.3), 
        height=0.15
    )

    # --- 2. Initialize Variables ---
    timer = core.Clock()
    familiarity_rating = 0
    liking_rating = 0
    
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
        keys = event.getKeys(keyList=['1', '2', '3', '4', '5', 
                                      'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 
                                      'escape'])
        
        if 'escape' in keys:
            window.close()
            core.quit()
            
        if keys:
            key_pressed = keys[0]
            if key_pressed.startswith('num_'):
                familiarity_rating = int(key_pressed.split('_')[1])
            else:
                familiarity_rating = int(key_pressed)
            
            # Show feedback
            feedback_stim.setText(str(familiarity_rating))
            fam_question.draw()
            for label in scale_labels:
                label.draw()
            feedback_stim.draw()
            window.flip()
            core.wait(config.FEEDBACK_DISPLAY_DURATION)
            break

    # --- 4. Run Liking Question Loop ---
    event.clearEvents()
    timer.reset()
    while timer.getTime() < config.LIKING_RESPONSE_DURATION:
        # Draw question and scale
        lik_question.draw()
        for label in scale_labels:
            label.draw()
        window.flip()

        # Check for keypress
        keys = event.getKeys(keyList=['1', '2', '3', '4', '5', 
                                      'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 
                                      'escape'])
        
        if 'escape' in keys:
            window.close()
            core.quit()
            
        if keys:
            key_pressed = keys[0]
            if key_pressed.startswith('num_'):
                liking_rating = int(key_pressed.split('_')[1])
            else:
                liking_rating = int(key_pressed)
            
            # Show feedback
            feedback_stim.setText(str(liking_rating))
            lik_question.draw()
            for label in scale_labels:
                label.draw()
            feedback_stim.draw()
            window.flip()
            core.wait(config.FEEDBACK_DISPLAY_DURATION)
            break

    return (familiarity_rating, liking_rating)