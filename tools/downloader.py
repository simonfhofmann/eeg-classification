import os
import sys
from yt_dlp import YoutubeDL

def download_fixed_segment(song_titles, participant_name, output_dir="downloaded_songs"):
    """
    Downloads a 32-second segment (seconds 20-52) of each YouTube song as WAV.
    
    song_titles: list of song names
    participant_name: string, folder will be created with this name
    """
    base_path = os.path.join(output_dir, participant_name)
    os.makedirs(base_path, exist_ok=True)

    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(base_path, '%(title)s.%(ext)s'),
        
        # 1. Force FFmpeg to cut the audio AFTER download but DURING conversion.
        # -ss 20: Start at 20 seconds
        # -t 32:  Last for 32 seconds (Total: 20s to 52s)
        'postprocessor_args': ['-ss', '20', '-t', '32'],
        
        # 2. Convert to WAV
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        
        'extractaudio': True,
        'audioformat': 'wav',
        'quiet': False,
        'no_warnings': False,
        'continuedl': True,
        'geo_bypass': True,
        # We remove 'download_sections' because it can be unreliable for certain formats.
        # The postprocessor_args above are the robust way to do it.
    }

    with YoutubeDL(ydl_opts) as ydl:
        for i, song in enumerate(song_titles, 1):
            # 'ytsearch1' grabs the first result. Adding "lyrics" helps find the studio version.
            query = f"ytsearch1:{song} lyrics" 
            print(f"\n[{i}/{len(song_titles)}] Searching for: {song}")
            
            try:
                ydl.download([query])
                print(f"   → Success: Saved 32s segment (20s-52s)")
            except Exception as e:
                print(f"   ✗ Failed: {e}", file=sys.stderr)

    print(f"\nAll done! Files are in: {os.path.abspath(base_path)}")


# ==================== YOUR LIST HERE ====================
# songs = [
#     "Olivia Rodrigo - Good 4 u",
#     "Doja Cat - Kiss Me More",
#     "The Weeknd - Blinding Lights",
#     "Dua Lipa - Levitating",
#     "Ed Sheeran - Shape of You",
#     "Taylor Swift - Shake It Off",
#     "Harry Styles - As It Was",
#     "Billie Eilish - Bad Guy",
#     "Ariana Grande - Thank U, Next",
#     "Imagine Dragons - Believer",
#     "Lil Nas X - Old Town Road",
#     "Justin Bieber - Sorry",
#     "Miley Cyrus - Flowers",
#     "SZA - Kill Bill",
#     "Mark Ronson ft. Bruno Mars - Uptown Funk",
#     "Twenty One Pilots - Stressed Out",
#     "Post Malone - Sunflower",
#     "Glass Animals - Heat Waves",
#     "Lewis Capaldi - Someone You Loved",
#     "One Direction - What Makes You Beautiful",
#     "BTS - Dynamite",
#     "Adele - Hello",
#     "Bruno Mars - Uptown Funk",
#     "Queen - Bohemian Rhapsody",
#     "Avicii - Wake Me Up",
#     "Britney Spears - ...Baby One More Time",
#     "Lady Gaga - Shallow",
#     "OneRepublic - Counting Stars",
#     "The Chainsmokers ft. Halsey - Closer",
#     "Eminem - Lose Yourself",
#     "The Killers - Mr. Brightside",
#     "Tones And I - Dance Monkey",
#     "Shawn Mendes & Camila Cabello - Señorita",
#     "Hozier - Take Me to Church",
#     "Coldplay - Viva La Vida"
# ]

# participant = "popular"
songs = [
    "KennyHoopla - estella//",
    "The Band CAMINO - Daphne Blue",
    "JID - 151 Rum",
    "The Midnight - Days of Thunder",
    "Royal Blood - Typhoons",
    "COIN - Crash My Car",
    "Denzel Curry - Walkin",
    "Des Rocs - Let Me Live / Let Me Die",
    "Smino - Wild Irish Roses",
    "Jungle - Keep Moving",
    "Cleopatrick - Hometown",
    "Badflower - Ghost",
    "Duckwrth - MICHUUL.",
    "Barns Courtney - Glitter & Gold",
    "The Blue Stones - Black Holes",
    "IDK - Digital",
    "Valley - Like 1999",
    "Don Broco - T-Shirt Song",
    "Saba - Photosynthesis",
    "Grandson - Blood // Water",
    "Welshly Arms - Legendary",
    "Kota the Friend - Colorado",
    "The Score - Unstoppable",
    "Nightly - The Movies",
    "Kenny Mason - Hit",
    "Saint Motel - Move",
    "Bakar - Hell N Back",
    "EarthGang - Meditate",
    "MISSIO - Middle Fingers",
    "Oliver Tree - Alien Boy",
    "Dreamers - Sweet Disaster",
    "6LACK - PRBLMS",
    "YUNGBLUD - parents",
    "Two Door Cinema Club - What You Know",
    "Aminé - REEL IT IN"
]
participant = "lesser_known"
if __name__ == "__main__":
    download_fixed_segment(songs, participant)