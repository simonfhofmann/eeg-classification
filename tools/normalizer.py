import os
from pydub import AudioSegment

# -----------------------------------------
# Your input path
# -----------------------------------------
INPUT_FOLDER = r"C:\Users\simon\Documents\master\25WS\praktikum_eeg\code\eeg-classification\downloaded_songs\lesser_known"

# Automatically create output folder next to input
parent = os.path.dirname(INPUT_FOLDER)
folder_name = os.path.basename(INPUT_FOLDER)
OUTPUT_FOLDER = os.path.join(parent, folder_name + "_normalized")

# Target loudness for all files
TARGET_LOUDNESS = -20.0  # dBFS
# -----------------------------------------


def match_target_amplitude(sound, target_dBFS):
    """Normalize audio to a target dBFS."""
    change = target_dBFS - sound.dBFS
    return sound.apply_gain(change)


def normalize_folder(input_folder, output_folder, target_loudness):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"Processing: {filename}")

        # Load audio
        audio = AudioSegment.from_file(input_path)

        # Normalize loudness
        normalized = match_target_amplitude(audio, target_loudness)

        # Export normalized file
        normalized.export(output_path, format=filename.split(".")[-1])

        print(f"Saved: {output_path}")

    print("\nAll files normalized successfully!\n")


if __name__ == "__main__":
    normalize_folder(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_LOUDNESS)
