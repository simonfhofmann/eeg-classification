# ----------------------------------------------------------------------
# eeg_handler.py
#
# Manages the hardware connection for sending EEG event markers
# via the parallel port (Linux only for live mode).
# ----------------------------------------------------------------------

from psychopy import parallel
import config

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

        if config.DEBUG_MODE:
            # --- DEBUG MODE ---
            self.port = self._DummyPort()
            print("-------------------------------------------------")
            print("---            DEBUG MODE ACTIVE              ---")
            print("--- No parallel port connection will be made. ---")
            print("--- Markers will be printed to the console.   ---")
            print("-------------------------------------------------")
        else:
            # --- LIVE MODE (Linux) ---
            try:
                self.port = parallel.ParallelPort(address=self.port_address)
                self.port.setData(0)
                print(f"Successfully connected to parallel port at {self.port_address}")
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to connect to parallel port at {self.port_address}.")
                print("Possible causes:")
                print("  1. Port address is incorrect (check with 'ls /dev/parport*')")
                print("  2. Insufficient permissions (try: sudo chmod 666 /dev/parport*)")
                print("  3. Parallel port driver not loaded (try: sudo modprobe parport_pc)")
                print(f"Details: {e}")
                raise

    def send_marker(self, value):
        """
        Sends an integer marker to the EEG amplifier (or dummy port).
        
        Args:
            value (int): Marker value to send (1-255)
        """
        if self.port is None:
            print(f"Error: Port not initialized. Cannot send marker {value}.")
            return
        
        # Validate marker value
        if not isinstance(value, int) or value < 0 or value > 255:
            print(f"Warning: Invalid marker value {value}. Must be integer 0-255.")
            return
            
        try:
            if config.DEBUG_MODE:
                self.port.setData(value)
            else:
                print(f"Sending Marker: {value}")
                self.port.setData(value)
                
        except Exception as e:
            print(f"Error sending marker {value}")
            print(f"Details: {e}")