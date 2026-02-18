import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import json
import os
import re

# --- CONFIGURATION ---
CACHE_DIR = '../fastf1_cache' # Relative path from streamlit_app/
DNA_FILE = '../coefficient/track_dna.json'
OUTPUT_DIR = 'assets/track_maps'
YEAR_FOR_MAP = 2024 # Use 2024 maps for most accuracy

def setup():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Enable Cache
    if os.path.exists(CACHE_DIR):
        fastf1.Cache.enable_cache(CACHE_DIR)
    
    # Setup FastF1 plotting (removes axis, etc.)
    fastf1.plotting.setup_mpl(misc_mpl_mods=False)

def clean_filename(track_name):
    """Converts 'Bahrain Grand Prix' to 'bahrain'"""
    # Remove 'Grand Prix', lower case, replace spaces with underscores if needed
    name = track_name.replace(" Grand Prix", "").strip().lower()
    return re.sub(r'[^a-z0-9]', '', name) # specific -> 'saudiarabian', 'bahrain'

def generate_maps():
    setup()
    
    # Load Track List
    with open(DNA_FILE, 'r') as f:
        data = json.load(f)
        tracks = [k for k in data.keys() if k != "_metadata"]

    print(f"🎨 Generating {len(tracks)} track maps...")

    for track_name in tracks:
        file_name = clean_filename(track_name)
        save_path = os.path.join(OUTPUT_DIR, f"{file_name}.svg")
        
        if os.path.exists(save_path):
            print(f"⏩ Skipping {file_name} (already exists)")
            continue

        try:
            # We use Qualifying for fastest clean lap
            session = fastf1.get_session(YEAR_FOR_MAP, track_name, 'Q')
            session.load()
            lap = session.laps.pick_fastest()
            
            # Get Coordinates
            pos = lap.get_pos_data()
            x = pos['X'].values
            y = pos['Y'].values
            
            # --- PLOTTING ---
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 1. The Glow (Thick, Low Opacity)
            ax.plot(x, y, color='#FF1801', linewidth=10, alpha=0.3, zorder=1)
            
            # 2. The Core Line (Thin, Solid)
            ax.plot(x, y, color='#FF1801', linewidth=3, alpha=1.0, zorder=2)
            
            ax.axis('off')
            ax.set_aspect('equal')
            
            # Save
            plt.savefig(save_path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"✅ Generated {file_name}.svg")
            
        except Exception as e:
            print(f"❌ Failed to generate {track_name}: {e}")

if __name__ == "__main__":
    generate_maps()