# ----------------------------------------------------------------------
# eeg_handler.py
#
# Manages the hardware connection for sending EEG event markers
# via the parallel port.
# ----------------------------------------------------------------------

from psychopy import parallel
import time

class MarkerHandler:
    def __init__(self, port_address):
        """
        Initializes the connection to the parallel port.

        Args:
            port_address (int): The memory address (e.g., 0xDC00)
                                of your parallel port.
        """
        self.port = None
        self.port_address = port_address
        try:
            self.port = parallel.ParallelPort(address=self.port_address)
            # Send a '0' marker immediately to clear the port
            self.port.setData(0)
            print(f"Successfully connected to parallel port at {hex(self.port_address)}")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to connect to parallel port at {hex(self.port_address)}.")
            print("Check port address in config.py. Is the 'inpoutx64.dll' library installed?")
            print(f"Details: {e}")
            raise

    def send_marker(self, value):
        """
        Sends an integer marker to the EEG amplifier.
        Resets the port to 0 after a very short delay.

        Args:
            value (int): The integer marker to send (0-255).
        """
        if self.port is None:
            print(f"Error: Port not initialized. Cannot send marker {value}.")
            return
            
        try:
            # print(f"Sending marker: {value}") # Uncomment for debugging
            self.port.setData(value)
            
            # Brief delay to ensure the amplifier registers the pulse
            time.sleep(0.005) 
            
            self.port.setData(0)
        except Exception as e:
            print(f"Error sending marker {value} to port {hex(self.port_address)}")
            print(f"Details: {e}")