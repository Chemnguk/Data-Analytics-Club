import pandas as pd
import numpy as np
import os
from glob import glob

base_dir = "/kaggle/input/ariel-data-challenge-2025/train"
planet_ids = sorted(os.listdir(base_dir), key=lambda x: int(x))

# === Function to extract and save signal features ===
def extract_signal_data(instrument):
    adc_df = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/adc_info.csv")
    gain = adc_df[f"{instrument}_adc_gain"].iloc[0]
    offset = adc_df[f"{instrument}_adc_offset"].iloc[0]

    features = []
    used_ids = []

    for pid in planet_ids:
        signal_files = glob(f"{base_dir}/{pid}/{instrument}_signal_*.parquet")
        mean_rows = []

        for path in signal_files:
            try:
                arr = pd.read_parquet(path).values.astype('float32')
                arr = arr * gain + offset
                mean_rows.append(np.mean(arr, axis=0))
            except:
                continue

        if mean_rows:
            features.append(np.mean(mean_rows, axis=0))
            used_ids.append(int(pid))

    df = pd.DataFrame(features, index=used_ids)
    df.to_parquet(f"{instrument.lower()}_signal_all.parquet")
    return df

# === Extract and save AIRS-CH0 and FGS1 signal data ===
airs_signal = extract_signal_data("AIRS-CH0")
fgs_signal = extract_signal_data("FGS1")

print("✅ AIRS-CH0 and FGS1 signals saved.")

# === General function to extract and save calibration frames ===
def extract_calibration_data(instrument, calib_type):
    all_frames = []
    used_planets = []

    for pid in planet_ids:
        path = os.path.join(base_dir, pid, f"{instrument}_calibration_0", f"{calib_type}.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path).values.flatten()
                all_frames.append(df)
                used_planets.append(int(pid))
            except:
                continue

    df = pd.DataFrame(all_frames, index=used_planets)
    df.to_parquet(f"{instrument.lower()}_{calib_type}_all.parquet")
    return df

# === Extract and save AIRS-CH0 calibration files ===
airs_dark = extract_calibration_data("AIRS-CH0", "dark")
airs_flat = extract_calibration_data("AIRS-CH0", "flat")
airs_dead = extract_calibration_data("AIRS-CH0", "dead")
airs_linear = extract_calibration_data("AIRS-CH0", "linear_corr")
airs_read = extract_calibration_data("AIRS-CH0", "read")

# === Extract and save FGS1 calibration files ===
fgs_dark = extract_calibration_data("FGS1", "dark")
fgs_flat = extract_calibration_data("FGS1", "flat")
fgs_dead = extract_calibration_data("FGS1", "dead")
fgs_linear = extract_calibration_data("FGS1", "linear_corr")
fgs_read = extract_calibration_data("FGS1", "read")

print("✅ AIRS-CH0 and FGS1 calibration files saved.")