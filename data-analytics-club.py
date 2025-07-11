import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

# === Paths ===
train_path = "/kaggle/input/ariel-data-challenge-2025/train"
adc_info_path = "/kaggle/input/ariel-data-challenge-2025/adc_info.csv"

# === Load ADC correction parameters ===
adc = pd.read_csv(adc_info_path)
gain = adc['AIRS-CH0_adc_gain'].iloc[0]
offset = adc['AIRS-CH0_adc_offset'].iloc[0]

# === Get all planet_ids ===
planet_ids = sorted(os.listdir(train_path), key=lambda x: int(x))

# === Final features container ===
features = []
planet_ids_used = []

# === Process each planet ===
for pid in tqdm(planet_ids, desc="Extracting AIRS-CH0 features"):
    signal_files = glob(os.path.join(train_path, pid, 'AIRS-CH0_signal_*.parquet'))

    planet_feature_rows = []
    for file_path in signal_files:
        df = pd.read_parquet(file_path).values.astype('float32')
        df = df * gain + offset

        # Example feature: mean flux across time (axis=0)
        mean_row = df.mean(axis=0)
        planet_feature_rows.append(mean_row)

    if planet_feature_rows:
        planet_feature = np.mean(planet_feature_rows, axis=0)  # avg if multiple observations
        features.append(planet_feature)
        planet_ids_used.append(pid)

# === Build DataFrame ===
airs_features_df = pd.DataFrame(features, index=planet_ids_used)
airs_features_df.index.name = 'planet_id'

# === Save to disk (optional) ===
airs_features_df.to_parquet("airs_features_summary.parquet")

print("Final feature shape:", airs_features_df.shape)