"""
01_load_data.py
---------------
Downloads and caches FastF1 telemetry data for ALL drivers across the full
2024 F1 season (all race rounds).
Outputs:
  data/laps.csv                – per-lap data for all drivers & races
  data/telemetry.csv           – raw telemetry for all laps
  data/telemetry_labelled.csv  – telemetry + Is_Anomaly column

Runtime: several hours on first run (downloads ~24 race caches, 20 drivers each).
Subsequent runs are fast thanks to the local cache.
"""

import os
import fastf1
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR = "cache"
DATA_DIR  = "data"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

YEAR = 2025

LAP_COLS = [
    "Driver", "DriverNumber",
    "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Compound", "TyreLife", "FreshTyre",
    "PitOutTime", "PitInTime",
    "TrackStatus", "IsPersonalBest",
    "Team",
]
TIMEDELTA_COLS = [
    "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
    "PitOutTime", "PitInTime",
]
TEL_COLS = ["Time", "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS",
            "X", "Y", "Z"]

# ── Discover all 2024 race rounds ─────────────────────────────────────────────
print(f"Fetching {YEAR} event schedule …")
schedule = fastf1.get_event_schedule(YEAR, include_testing=False)
race_rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()
print(f"  Found {len(race_rounds)} race rounds: {race_rounds}\n")

# ── Accumulate data across all rounds ─────────────────────────────────────────
all_laps      = []
all_telemetry = []

for rnd in race_rounds:
    event_name = schedule.loc[
        schedule["RoundNumber"] == rnd, "EventName"
    ].values[0]

    print(f"[Round {rnd:02d}] {event_name} …")

    try:
        session = fastf1.get_session(YEAR, rnd, "R")
        session.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception as e:
        print(f"  ⚠  Could not load session: {e}")
        continue

    # ── Laps (all drivers) ────────────────────────────────────────────────────
    all_driver_laps = session.laps.reset_index(drop=True)
    if all_driver_laps.empty:
        print(f"  ⚠  No laps found for this session")
        continue

    drivers = all_driver_laps["Driver"].unique()
    print(f"  Drivers: {sorted(drivers)}")

    laps_df = all_driver_laps[[c for c in LAP_COLS if c in all_driver_laps.columns]].copy()

    for col in TIMEDELTA_COLS:
        if col in laps_df.columns:
            laps_df[col] = laps_df[col].dt.total_seconds()

    laps_df.insert(0, "Round",     rnd)
    laps_df.insert(1, "EventName", event_name)
    all_laps.append(laps_df)
    print(f"  Total laps: {len(laps_df)}")

    # ── Telemetry (all drivers) ───────────────────────────────────────────────
    race_tel_frames = []
    for drv in drivers:
        driver_laps = all_driver_laps[all_driver_laps["Driver"] == drv]
        drv_count = 0
        for _, lap in driver_laps.iterrows():
            try:
                tel = lap.get_telemetry()
                avail = [c for c in TEL_COLS if c in tel.columns]
                tel   = tel[avail].copy()
                tel["Driver"]    = drv
                tel["LapNumber"] = lap["LapNumber"]
                tel["LapTime_s"] = (
                    lap["LapTime"].total_seconds()
                    if pd.notna(lap["LapTime"]) else np.nan
                )
                race_tel_frames.append(tel)
                drv_count += 1
            except Exception:
                pass  # silently skip laps with no telemetry
        if drv_count == 0:
            print(f"    ⚠  No telemetry for {drv}")

    if race_tel_frames:
        race_tel = pd.concat(race_tel_frames, ignore_index=True)
        if "Time" in race_tel.columns:
            race_tel["Time"] = race_tel["Time"].dt.total_seconds()
        race_tel.insert(0, "Round",     rnd)
        race_tel.insert(1, "EventName", event_name)
        all_telemetry.append(race_tel)
        print(f"  Telemetry rows: {len(race_tel):,}")
    else:
        print(f"  ⚠  No telemetry extracted for this round")

# ── Concatenate & save raw data ───────────────────────────────────────────────
print("\nConcatenating all rounds …")

laps_all = pd.concat(all_laps, ignore_index=True)
laps_all.to_csv(f"{DATA_DIR}/laps.csv", index=False)
print(f"  Saved {DATA_DIR}/laps.csv  "
      f"({len(laps_all):,} rows | {laps_all['Round'].nunique()} rounds | "
      f"{laps_all['Driver'].nunique()} drivers)")

telemetry_all = pd.concat(all_telemetry, ignore_index=True)
telemetry_all.to_csv(f"{DATA_DIR}/telemetry.csv", index=False)
print(f"  Saved {DATA_DIR}/telemetry.csv  ({len(telemetry_all):,} rows)")

# ── Anomaly labelling (per-race, per-driver z-score on LapTime) ───────────────
# A lap is anomalous if:
#   (a) LapTime > driver_race_mean + 1.5 * driver_race_std, OR
#   (b) TrackStatus != '1'  (safety car, VSC, red flag)
print("\nLabelling anomalies …")

anomaly_keys = set()  # (Round, Driver, LapNumber) tuples

for (rnd, drv), grp in laps_all.groupby(["Round", "Driver"]):
    lap_times = grp["LapTime"].dropna()
    if len(lap_times) < 3:
        continue
    thr = lap_times.mean() + 1.5 * lap_times.std()

    for ln in grp.loc[grp["LapTime"] > thr, "LapNumber"]:
        anomaly_keys.add((rnd, drv, ln))

    if "TrackStatus" in grp.columns:
        for ln in grp.loc[grp["TrackStatus"] != "1", "LapNumber"]:
            anomaly_keys.add((rnd, drv, ln))

telemetry_labelled = telemetry_all.copy()
telemetry_labelled["Is_Anomaly"] = telemetry_labelled.apply(
    lambda row: int((row["Round"], row["Driver"], row["LapNumber"]) in anomaly_keys),
    axis=1,
)

n_anomaly = int(telemetry_labelled["Is_Anomaly"].sum())
n_total   = len(telemetry_labelled)
print(f"  Anomaly samples : {n_anomaly:,} / {n_total:,} "
      f"({100 * n_anomaly / n_total:.2f} %)")

telemetry_labelled.to_csv(f"{DATA_DIR}/telemetry_labelled.csv", index=False)
print(f"  Saved {DATA_DIR}/telemetry_labelled.csv")

# ── Season summary ────────────────────────────────────────────────────────────
print("\n── Season summary (laps per driver) ─────────────────────────────────")
summary = (
    laps_all.groupby("Driver")
    .agg(
        Rounds      =("Round",    "nunique"),
        TotalLaps   =("LapNumber","count"),
        AvgLapTime_s=("LapTime",  "mean"),
    )
    .sort_values("TotalLaps", ascending=False)
    .reset_index()
)
print(summary.to_string(index=False))

print("\n── Laps per round ───────────────────────────────────────────────────")
round_summary = (
    laps_all.groupby(["Round", "EventName"])
    .agg(
        Drivers  =("Driver",    "nunique"),
        TotalLaps=("LapNumber", "count"),
    )
    .reset_index()
)
print(round_summary.to_string(index=False))

print("\n✅  01_load_data.py complete.")