"""
01_load_data.py
FastF1 Project – Data Loading & Caching
----------------------------------------
Loads telemetry, lap, and weather data for a given session using the
FastF1 library and saves them as CSV files for downstream scripts.

Usage:
    python 01_load_data.py
"""

import fastf1
import pandas as pd
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
YEAR     = 2020
RACE     = "Silverstone"
SESSION  = "R"          # 'R' = Race, 'Q' = Qualifying, 'FP1/FP2/FP3' = Practice
DRIVER   = "HAM"        # 3-letter driver code
CACHE_DIR  = Path("cache")
OUTPUT_DIR = Path("data")

# ── Setup ─────────────────────────────────────────────────────────────────────
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Load Session ──────────────────────────────────────────────────────────────
print(f"Loading {YEAR} {RACE} – Session: {SESSION} …")
session = fastf1.get_session(YEAR, RACE, SESSION)
session.load()

# ── Laps ──────────────────────────────────────────────────────────────────────
driver_laps = session.laps.pick_drivers(DRIVER).copy()
driver_laps = driver_laps.reset_index(drop=True)

# Convert timedelta columns to seconds for easier analysis
time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
             "PitInTime", "PitOutTime"]
for col in time_cols:
    if col in driver_laps.columns:
        driver_laps[col] = driver_laps[col].dt.total_seconds()

print(f"  Laps loaded   : {len(driver_laps)} laps")
driver_laps.to_csv(OUTPUT_DIR / "laps.csv", index=False)
print(f"  Saved → {OUTPUT_DIR}/laps.csv")

# ── Telemetry ─────────────────────────────────────────────────────────────────
# Collect telemetry for every lap and tag each row with lap number
telemetry_frames = []
for _, lap in driver_laps.iterrows():
    try:
        tel = lap.get_car_data().add_distance()
        tel["LapNumber"] = lap["LapNumber"]
        tel["LapTime_s"] = lap["LapTime"]   # already in seconds
        telemetry_frames.append(tel)
    except Exception:
        pass  # skip laps with missing telemetry

telemetry = pd.concat(telemetry_frames, ignore_index=True)

# Drop rows with any NA in core sensor columns
core_cols = ["Speed", "RPM", "nGear", "Throttle", "Brake", "DRS"]
telemetry.dropna(subset=core_cols, inplace=True)

print(f"  Telemetry rows: {len(telemetry):,}")
telemetry.to_csv(OUTPUT_DIR / "telemetry.csv", index=False)
print(f"  Saved → {OUTPUT_DIR}/telemetry.csv")

# ── Weather ───────────────────────────────────────────────────────────────────
weather = session.weather_data.copy()
if weather is not None and not weather.empty:
    weather.to_csv(OUTPUT_DIR / "weather.csv", index=False)
    print(f"  Weather rows  : {len(weather)}")
    print(f"  Saved → {OUTPUT_DIR}/weather.csv")
else:
    print("  No weather data available for this session.")

print("\n✓ Data loading complete.")