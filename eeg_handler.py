# ----------------------------------------------------------------------
# eeg_handler.py
#
# Manages the hardware connection for sending EEG event markers
# via the parallel port.
# ----------------------------------------------------------------------

from psychopy import parallel
import time
import config  # <-- 1. IMPORT CONFIG

class MarkerHandler:
    
    class _DummyPort:
        """
        A fake port object that mimics the 'setData' method
        for debugging. It just prints to the console.
        """
        def setData(self, value):
            if value != 0: 
                print(f"--- MARKER SENT (DEBUG): {value} ---")

    
    def __init__(self, port_address):
        """
        Initializes the connection to the parallel port OR
        creates a dummy port if in DEBUG_MODE.
        """
        self.port = None
        self.port_address = port_address

        # --- 2. CHECK THE DEBUG FLAG ---
        if config.DEBUG_MODE:
            # --- DEBUG MODE ---
            self.port = self._DummyPort()
            print("-------------------------------------------------")
            print("---            DEBUG MODE ACTIVE            ---")
            print("--- No parallel port connection will be made. ---")
            print("--- Markers will be printed to the console.   ---")
            print("-------------------------------------------------")
        else:
            # --- LIVE MODE ---
            try:
                self.port = parallel.ParallelPort(address=self.port_address)
                # Send a '0' marker immediately to clear the port
                self.port.setData(0)
                print(f"Successfully connected to parallel port at {hex(self.port_address)}")
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to connect to parallel port at {hex(self.port_address)}.")
                print("Check port address. Is the 'inpoutx64.dll' library installed?")
                print(f"Details: {e}")
                raise

    def send_marker(self, value):
        """
        Sends an integer marker to the EEG amplifier (or dummy port).
        """
        if self.port is None:
            print(f"Error: Port not initialized. Cannot send marker {value}.")
            return
            
        try:
            # --- 3. SEND MARKER (conditionally) ---
            if config.DEBUG_MODE:
                self.port.setData(value)
            else:
                self.port.setData(value)
                time.sleep(0.005) # Hardware pulse duration
                self.port.setData(0) # Reset port
        except Exception as e:
            print(f"Error sending marker {value}")
            print(f"Details: {e}")