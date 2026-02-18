import fastf1
import json
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm  # pip install tqdm if not installed

# --- CONFIGURATION ---
CACHE_DIR = 'fastf1_cache'  # Pointing to your existing cache folder
DNA_FILE = 'coefficient/track_dna.json'
OUTPUT_FILE = 'coefficient/historical_podiums.json'
YEARS_TO_FETCH = [2024, 2023, 2022, 2021, 2020]

def setup():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Enable FastF1 Cache
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    fastf1.Cache.enable_cache(CACHE_DIR)

def get_track_list():
    """Reads the track_dna.json to get the official list of tracks."""
    if not os.path.exists(DNA_FILE):
        print(f"❌ Error: {DNA_FILE} not found.")
        return []
    
    with open(DNA_FILE, 'r') as f:
        data = json.load(f)
        # Filter out metadata keys
        tracks = [k for k in data.keys() if k != "_metadata"]
    return tracks

def fetch_podiums():
    setup()
    tracks = get_track_list()
    historical_data = {}

    print(f"📊 Fetching podiums for {len(tracks)} tracks across {len(YEARS_TO_FETCH)} years...")

    for track_name in tqdm(tracks, desc="Processing Tracks"):
        track_podiums = []
        
        # Clean name for FastF1 (e.g., "Bahrain Grand Prix" -> "Bahrain")
        # FastF1 is smart, but providing the country/location often works best
        # We'll try using the full name first, as FastF1 usually handles "Grand Prix" suffix well.
        
        for year in YEARS_TO_FETCH:
            try:
                # Load the Race Session
                session = fastf1.get_session(year, track_name, 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                
                # Get Top 3
                results = session.results.iloc[:3]
                
                podium_entry = {
                    "year": year,
                    "drivers": results['Abbreviation'].tolist() # ['VER', 'HAM', 'LEC']
                }
                track_podiums.append(podium_entry)
                
            except Exception as e:
                # Silent fail for cancelled races (e.g., Imola 2023) or missing data
                continue
        
        # Consolidate data: Count total podiums per driver for this track
        driver_counts = {}
        for entry in track_podiums:
            for driver in entry['drivers']:
                driver_counts[driver] = driver_counts.get(driver, 0) + 1
        
        # Sort by most podiums
        sorted_counts = dict(sorted(driver_counts.items(), key=lambda item: item[1], reverse=True))
        
        historical_data[track_name] = sorted_counts

    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(historical_data, f, indent=2)
    
    print(f"✅ Success! Historical data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_podiums()