"""
01_load_data.py
---------------
Downloads and caches FastF1 data for ALL drivers across the full 2025 F1
season (Race sessions only).

Outputs
-------
  data/laps.csv             – per-lap data (enhanced) + weather merged by time
  data/telemetry.csv        – raw telemetry for all laps
  data/telemetry_labelled.csv – telemetry + Is_Anomaly column
  data/weather.csv          – raw weather time-series per round
  data/results.csv          – race results + driver info per round
  data/race_control.csv     – race control messages per round

Memory strategy
---------------
Data is written to CSV incrementally after each round (append mode) so RAM
usage is bounded to one session at a time instead of the full season (~2 GB).
The telemetry loop is intentionally separate from the laps/weather/results loop
so either can be re-run independently if a round fails mid-way.
Anomaly labelling reads telemetry in 500k-row chunks so it never loads 2 GB
at once.

What FastF1 does NOT provide:
  - Car setups (confidential, team-only)
  - Tyre temperatures / pressures from sensors
  - Fuel loads / engine modes

Runtime: several hours on first run; fast on re-runs thanks to the local cache.
"""

import os, gc
import fastf1
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR = "../cache"
DATA_DIR  = "../data"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

YEAR = 2025

# ── Column definitions ────────────────────────────────────────────────────────
LAP_COLS = [
    "Driver", "DriverNumber", "Team",
    "LapNumber", "LapTime",
    "Sector1Time", "Sector2Time", "Sector3Time",
    "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    "LapStartTime", "LapStartDate",
    "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "Compound", "TyreLife", "FreshTyre", "Stint",
    "PitOutTime", "PitInTime",
    "Position", "TrackStatus", "IsPersonalBest",
    "IsAccurate", "Deleted", "DeletedReason",
]
TIMEDELTA_COLS = [
    "LapTime",
    "Sector1Time", "Sector2Time", "Sector3Time",
    "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    "LapStartTime", "PitOutTime", "PitInTime",
]
TEL_COLS = ["Time", "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS",
            "X", "Y", "Z"]
WEATHER_COLS = ["Time", "AirTemp", "Humidity", "Pressure",
                "Rainfall", "TrackTemp", "WindDirection", "WindSpeed"]
RESULT_COLS = [
    "DriverNumber", "Abbreviation", "FullName", "TeamName", "TeamId",
    "GridPosition", "Position", "ClassifiedPosition",
    "Points", "Status", "Laps", "Q1", "Q2", "Q3", "Time", "CountryCode",
]

# ── Discover all 2025 race rounds ─────────────────────────────────────────────
print(f"Fetching {YEAR} event schedule …")
schedule    = fastf1.get_event_schedule(YEAR, include_testing=False)
race_rounds = schedule[schedule["EventFormat"] != "testing"]["RoundNumber"].tolist()
print(f"  Found {len(race_rounds)} race rounds: {race_rounds}\n")

# ── Incremental CSV helper ────────────────────────────────────────────────────
def append_csv(df: pd.DataFrame, path: str, first: bool) -> None:
    """Write with header on first call; append without header on subsequent calls."""
    df.to_csv(path, index=False, mode="w" if first else "a", header=first)

# ══════════════════════════════════════════════════════════════════════════════
# LOOP 1 – Laps, weather, results, race control
# telemetry=False keeps this loop fast and light on RAM.
# Each round is saved immediately; a crash loses at most one round.
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("LOOP 1 — Laps / Weather / Results / Race control")
print("=" * 65)

first_round = True
completed   = []

for rnd in race_rounds:
    event_name = schedule.loc[schedule["RoundNumber"] == rnd, "EventName"].values[0]
    circuit    = schedule.loc[schedule["RoundNumber"] == rnd, "Location"].values[0]
    country    = schedule.loc[schedule["RoundNumber"] == rnd, "Country"].values[0]
    print(f"\n[Round {rnd:02d}] {event_name}  ({circuit}, {country})")

    try:
        session = fastf1.get_session(YEAR, rnd, "R")
        session.load(telemetry=False, laps=True, weather=True, messages=True)
    except Exception as e:
        print(f"  ⚠  Could not load session: {e}")
        continue

    # ── Results ───────────────────────────────────────────────────────────────
    try:
        res = session.results
        if res is not None and not res.empty:
            res_df = res[[c for c in RESULT_COLS if c in res.columns]].copy()
            for tc in ["Q1", "Q2", "Q3", "Time"]:
                if tc in res_df.columns:
                    res_df[tc] = res_df[tc].apply(
                        lambda x: x.total_seconds()
                        if pd.notna(x) and hasattr(x, "total_seconds") else np.nan
                    )
            res_df.insert(0, "Round",     rnd)
            res_df.insert(1, "EventName", event_name)
            res_df.insert(2, "Circuit",   circuit)
            res_df.insert(3, "Country",   country)
            append_csv(res_df, f"{DATA_DIR}/results.csv", first_round)
            print(f"  Results   : {len(res_df)} drivers")
    except Exception as e:
        print(f"  ⚠  Results error: {e}")

    # ── Weather ───────────────────────────────────────────────────────────────
    weather_df_raw = None
    weather_ok     = False
    try:
        wd = session.weather_data
        if wd is not None and not wd.empty:
            weather_df_raw = wd[[c for c in WEATHER_COLS if c in wd.columns]].copy()
            weather_df_raw["Time"] = weather_df_raw["Time"].dt.total_seconds()
            weather_df_raw.insert(0, "Round",     rnd)
            weather_df_raw.insert(1, "EventName", event_name)
            append_csv(weather_df_raw, f"{DATA_DIR}/weather.csv", first_round)
            weather_ok = True
            print(f"  Weather   : {len(weather_df_raw)} rows")
    except Exception as e:
        print(f"  ⚠  Weather error: {e}")

    # ── Race control messages ─────────────────────────────────────────────────
    try:
        rc = session.race_control_messages
        if rc is not None and not rc.empty:
            rc_df = rc.copy()
            rc_df["Time"] = rc_df["Time"].dt.total_seconds()
            rc_df.insert(0, "Round",     rnd)
            rc_df.insert(1, "EventName", event_name)
            append_csv(rc_df, f"{DATA_DIR}/race_control.csv", first_round)
            print(f"  Race ctrl : {len(rc_df)} messages")
    except Exception as e:
        print(f"  ⚠  Race control error: {e}")

    # ── Laps ──────────────────────────────────────────────────────────────────
    all_driver_laps = session.laps.reset_index(drop=True)
    if all_driver_laps.empty:
        print(f"  ⚠  No laps found")
        del session; gc.collect()
        continue

    laps_df = all_driver_laps[[c for c in LAP_COLS
                                if c in all_driver_laps.columns]].copy()
    for col in TIMEDELTA_COLS:
        if col in laps_df.columns:
            laps_df[col] = laps_df[col].dt.total_seconds()

    laps_df.insert(0, "Round",     rnd)
    laps_df.insert(1, "EventName", event_name)
    laps_df.insert(2, "Circuit",   circuit)
    laps_df.insert(3, "Country",   country)

    # Merge weather into laps by nearest session time
    if weather_ok and weather_df_raw is not None and "LapStartTime" in laps_df.columns:
        try:
            wcols  = [c for c in WEATHER_COLS
                      if c != "Time" and c in weather_df_raw.columns]
            wmerge = (weather_df_raw[["Time"] + wcols]
                      .sort_values("Time").reset_index(drop=True))
            laps_s = laps_df.sort_values("LapStartTime").reset_index()
            merged = pd.merge_asof(
                laps_s,
                wmerge.rename(columns={"Time": "LapStartTime"}),
                on="LapStartTime", direction="nearest",
            )
            laps_df = merged.set_index("index").sort_index().reset_index(drop=True)
        except Exception as e:
            print(f"  ⚠  Weather merge error: {e}")

    append_csv(laps_df, f"{DATA_DIR}/laps.csv", first_round)
    drivers = sorted(all_driver_laps["Driver"].unique())
    print(f"  Laps      : {len(laps_df)}  |  Drivers: {drivers}")

    first_round = False
    completed.append(rnd)
    del session, laps_df, all_driver_laps, weather_df_raw
    gc.collect()

print(f"\nLoop 1 complete — rounds saved: {completed}")

# ══════════════════════════════════════════════════════════════════════════════
# LOOP 2 – Telemetry
# Heavy: ~750 k rows per round. Saved incrementally so RAM holds one round
# at a time instead of the full ~18 M rows.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("LOOP 2 — Telemetry")
print("=" * 65)

first_tel = True

for rnd in race_rounds:
    event_name = schedule.loc[schedule["RoundNumber"] == rnd, "EventName"].values[0]
    print(f"\n[Round {rnd:02d}] {event_name} …")

    try:
        session = fastf1.get_session(YEAR, rnd, "R")
        session.load(telemetry=True, laps=True, weather=False, messages=False)
    except Exception as e:
        print(f"  ⚠  Could not load: {e}")
        continue

    all_driver_laps = session.laps.reset_index(drop=True)
    if all_driver_laps.empty:
        print(f"  ⚠  No laps")
        del session; gc.collect()
        continue

    drivers    = all_driver_laps["Driver"].unique()
    tel_frames = []

    for drv in drivers:
        drv_laps  = all_driver_laps[all_driver_laps["Driver"] == drv]
        drv_count = 0
        for _, lap in drv_laps.iterrows():
            try:
                tel   = lap.get_telemetry()
                avail = [c for c in TEL_COLS if c in tel.columns]
                tel   = tel[avail].copy()
                tel["Driver"]    = drv
                tel["LapNumber"] = lap["LapNumber"]
                tel["LapTime_s"] = (lap["LapTime"].total_seconds()
                                    if pd.notna(lap["LapTime"]) else np.nan)
                tel_frames.append(tel)
                drv_count += 1
            except Exception:
                pass
        if drv_count == 0:
            print(f"    ⚠  No telemetry for {drv}")

    if tel_frames:
        race_tel = pd.concat(tel_frames, ignore_index=True)
        if "Time" in race_tel.columns:
            race_tel["Time"] = race_tel["Time"].dt.total_seconds()
        race_tel.insert(0, "Round",     rnd)
        race_tel.insert(1, "EventName", event_name)
        append_csv(race_tel, f"{DATA_DIR}/telemetry.csv", first_tel)
        print(f"  Telemetry : {len(race_tel):,} rows")
        first_tel = False
        del race_tel
    else:
        print(f"  ⚠  No telemetry this round")

    del session, tel_frames, all_driver_laps
    gc.collect()

print("\nLoop 2 complete.")

# ══════════════════════════════════════════════════════════════════════════════
# Anomaly labelling
# ══════════════════════════════════════════════════════════════════════════════
# A lap is anomalous if:
#   (a) LapTime > driver_race_mean + 1.5 * std
#   (b) TrackStatus != '1'  (safety car, VSC, red flag)
#   (c) Deleted == True     (track limits violation)
# Telemetry is read in 500k-row chunks so we never load 2 GB at once.
print("\n" + "=" * 65)
print("Anomaly labelling")
print("=" * 65)

laps_all = pd.read_csv(f"{DATA_DIR}/laps.csv")
print(f"Loaded laps.csv : {len(laps_all):,} rows")

anomaly_keys = set()
for (rnd, drv), grp in laps_all.groupby(["Round", "Driver"]):
    lap_times = grp["LapTime"].dropna()
    if len(lap_times) >= 3:
        thr = lap_times.mean() + 1.5 * lap_times.std()
        for ln in grp.loc[grp["LapTime"] > thr, "LapNumber"]:
            anomaly_keys.add((rnd, drv, ln))
    if "TrackStatus" in grp.columns:
        for ln in grp.loc[grp["TrackStatus"] != "1", "LapNumber"]:
            anomaly_keys.add((rnd, drv, ln))
    if "Deleted" in grp.columns:
        for ln in grp.loc[grp["Deleted"] == True, "LapNumber"]:
            anomaly_keys.add((rnd, drv, ln))

print(f"Anomaly (Round, Driver, Lap) keys: {len(anomaly_keys):,}")

out_path    = f"{DATA_DIR}/telemetry_labelled.csv"
first_chunk = True
n_total = n_anomaly = 0

for chunk in pd.read_csv(f"{DATA_DIR}/telemetry.csv", chunksize=500_000):
    chunk["Is_Anomaly"] = chunk.apply(
        lambda row: int((row["Round"], row["Driver"], row["LapNumber"])
                        in anomaly_keys),
        axis=1,
    )
    n_anomaly += int(chunk["Is_Anomaly"].sum())
    n_total   += len(chunk)
    chunk.to_csv(out_path, index=False,
                 mode="w" if first_chunk else "a",
                 header=first_chunk)
    first_chunk = False

print(f"Anomaly rows : {n_anomaly:,} / {n_total:,} "
      f"({100 * n_anomaly / max(n_total, 1):.2f} %)")
print(f"Saved {out_path}")

# ══════════════════════════════════════════════════════════════════════════════
# Season summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Season summary (laps per driver) ─────────────────────────────────")
summary = (
    laps_all.groupby("Driver")
    .agg(Rounds=("Round", "nunique"),
         TotalLaps=("LapNumber", "count"),
         AvgLapTime_s=("LapTime", "mean"))
    .sort_values("TotalLaps", ascending=False)
    .reset_index()
)
print(summary.to_string(index=False))

print("\n── Laps per round ───────────────────────────────────────────────────")
round_summary = (
    laps_all.groupby(["Round", "EventName", "Circuit", "Country"])
    .agg(Drivers=("Driver", "nunique"),
         TotalLaps=("LapNumber", "count"))
    .reset_index()
)
print(round_summary.to_string(index=False))

print("\n── Lap columns saved ─────────────────────────────────────────────────")
print(sorted(laps_all.columns.tolist()))

print("\n✅  01_load_data.py complete.")
