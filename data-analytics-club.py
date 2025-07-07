# %% [code]
"""
created on 30-June-2025 19:49
author: ABIODUN TIAMIYU 
     
     TODAY'S AIM & LOGIC
GOAL:
Merge all AIRS-CH0 calibration files for every planet_id 
into a SINGLE large dataset by columns.
                                    
 The reason:
- Each planet_id folder contains calibration files inside AIRS-CH0_calibration_0.
- These files include:
    - dark.parquet
    - dead.parquet
    - flat.parquet
    - linear_corr.parquet
    - read.parquet

- All calibration files have similar shape (i.e., same number of rows).
- Instead of having thousands of separate small files, we want to:
    - Combine them side by side by columns.
    - Each planet's data gets unique column names that include its planet_id for easy tracking.

Final Dataset Structure Example:

dark_col_0_pid34983 | dark_col_1_pid34983 | ... | read_col_0_pid34983 | ... | dark_col_0_pid1873185 | ... | read_col_N_pid_last

Where:
- Rows = consistent across all calibration files (aligned vertically).
- Columns = all calibration data for all planets, neatly combined by column.
- Result = one giant dataframe ready for modeling or analysis.

Additional Notes:
-----------------
- To handle memory issues, batching by groups of 100 planet_ids is recommended.
- Each batch's merged dataframe can be saved separately and optionally merged later if memory allows.
"""
import os
import pandas as pd
from tqdm import tqdm

root_folder = "/kaggle/input/ariel-data-challenge-2025/train"
planet_ids = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
batch_size = 100

for batch_start in range(0, len(planet_ids), batch_size):
    batch_end = min(batch_start + batch_size, len(planet_ids))
    batch_ids = planet_ids[batch_start:batch_end]
    
    merged_batch_df = None
    
    for pid in tqdm(batch_ids):
        calib_dir = os.path.join(root_folder, pid, "AIRS-CH0_calibration_0")
        
        if not os.path.exists(calib_dir):
            continue
        
        planet_df_parts = []
        
        for file_name in ["dark.parquet", "dead.parquet", "flat.parquet", "linear_corr.parquet", "read.parquet"]:
            file_path = os.path.join(calib_dir, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    df.columns = [f"{file_name.replace('.parquet','')}_col{idx}_pid{pid}" for idx in range(df.shape[1])]
                    planet_df_parts.append(df)
                except:
                    continue
        
        if planet_df_parts:
            planet_merged_df = pd.concat(planet_df_parts, axis=1)
            merged_batch_df = pd.concat([merged_batch_df, planet_merged_df], axis=1) if merged_batch_df is not None else planet_merged_df
    
    if merged_batch_df is not None:
        output_path = f"airs_ch0_calibration_merged_batch_{batch_start // batch_size + 1}.parquet"
        merged_batch_df.to_parquet(output_path)
        
merged_df = pd.read_parquet("airs_ch0_calibration_merged_batch_1.parquet")
print(merged_df.shape)
print(merged_df.head())

#%% [code]
"""
Logic:
For all planet IDs inside the root folder:
- Locate the FGS1_calibration_0 folder for each planet.
- From each folder, read the calibration files: dark, dead, flat, linear_corr, read.
- Concatenate their columns horizontally for that planet.
- Repeat for all planet IDs in batches (to avoid memory overload).
- Save merged columns for each batch as a single .parquet file.
"""

# Root folder path
root_folder = "/kaggle/input/ariel-data-challenge-2025/train"  

# Get sorted planet IDs
planet_ids = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
batch_size = 100

def merge_fgs1_calibration_batch(batch_number):
    
    start_idx = batch_number * batch_size
    end_idx = min(start_idx + batch_size, len(planet_ids))
    batch_ids = planet_ids[start_idx:end_idx]

    merged_df = None

    print(f"\nMerging FGS1 Calibration Batch {batch_number + 1} with {len(batch_ids)} planet IDs...")

    for pid in tqdm(batch_ids):
        calib_dir = os.path.join(root_folder, pid, "FGS1_calibration_0")

        if not os.path.exists(calib_dir):
            continue

        planet_parts = []

        for file_name in ["dark.parquet", "dead.parquet", "flat.parquet", "linear_corr.parquet", "read.parquet"]:
            file_path = os.path.join(calib_dir, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_parquet(file_path)
                    df.columns = [f"{file_name.replace('.parquet','')}_col{idx}_pid{pid}" for idx in range(df.shape[1])]
                    planet_parts.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

        if planet_parts:
            planet_merged = pd.concat(planet_parts, axis=1)
            merged_df = pd.concat([merged_df, planet_merged], axis=1) if merged_df is not None else planet_merged

    if merged_df is not None:
        output_path = f"fgs1_merged_calibration_batch_{batch_number + 1}.parquet"
        merged_df.to_parquet(output_path)
        print(f"Saved merged batch to {output_path}, shape: {merged_df.shape}")

#%% [code]
"""
Logic:
For all planet IDs inside the root folder:
- Locate the AIRS-CH0_signal_0.parquet file for each planet.
- Read the file and rename columns to clearly reflect it's AIRS signal data and planet_id.
- Concatenate horizontally across all planet IDs in a batch.
- Save merged columns for each batch as a distinct .parquet file.
"""

# Root folder path
root_folder = "/kaggle/input/ariel-data-challenge-2025/train"  # Replace with your actual path

# Get sorted planet IDs
planet_ids = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
batch_size = 100

def merge_airs_signal_columns_batch(batch_number):
    
    start_idx = batch_number * batch_size
    end_idx = min(start_idx + batch_size, len(planet_ids))
    batch_ids = planet_ids[start_idx:end_idx]

    merged_airs_signal_df = None

    print(f"\nMerging AIRS-CH0 Signal Columns for Batch {batch_number + 1} with {len(batch_ids)} planet IDs...")

    for pid in tqdm(batch_ids):
        signal_path = os.path.join(root_folder, pid, "AIRS-CH0_signal_0.parquet")

        if not os.path.exists(signal_path):
            continue

        try:
            df = pd.read_parquet(signal_path)
            df.columns = [f"airs_sig_col{idx}_pid{pid}" for idx in range(df.shape[1])]
            merged_airs_signal_df = pd.concat([merged_airs_signal_df, df], axis=1) if merged_airs_signal_df is not None else df

        except Exception as e:
            print(f"Error reading {signal_path}: {e}")

    if merged_airs_signal_df is not None:
        output_path = f"merged_airs_signal_columns_batch_{batch_number + 1}.parquet"
        merged_airs_signal_df.to_parquet(output_path)
        print(f"Saved merged AIRS signal columns for batch to {output_path}, shape: {merged_airs_signal_df.shape}")

#%% [code]

"""
Logic:
For all planet IDs inside the root folder:
- Locate the FGS1_signal_0.parquet file for each planet.
- Read the file and rename columns to clearly reflect it's FGS1 signal data and planet_id.
- Concatenate horizontally across all planet IDs in a batch.
- Save merged columns for each batch as a distinct .parquet file.
"""

# Root folder path
root_folder = "/kaggle/input/ariel-data-challenge-2025/train"  

# Get sorted planet IDs
planet_ids = sorted([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))])
batch_size = 100

def merge_fgs1_signal_columns_batch(batch_number):
    
    start_idx = batch_number * batch_size
    end_idx = min(start_idx + batch_size, len(planet_ids))
    batch_ids = planet_ids[start_idx:end_idx]

    merged_fgs1_signal_df = None

    print(f"\nMerging FGS1 Signal Columns for Batch {batch_number + 1} with {len(batch_ids)} planet IDs...")

    for pid in tqdm(batch_ids):
        signal_path = os.path.join(root_folder, pid, "FGS1_signal_0.parquet")

        if not os.path.exists(signal_path):
            continue

        try:
            df = pd.read_parquet(signal_path)
            df.columns = [f"fgs1_sig_col{idx}_pid{pid}" for idx in range(df.shape[1])]
            merged_fgs1_signal_df = pd.concat([merged_fgs1_signal_df, df], axis=1) if merged_fgs1_signal_df is not None else df

        except Exception as e:
            print(f"Error reading {signal_path}: {e}")

    if merged_fgs1_signal_df is not None:
        output_path = f"merged_fgs1_signal_columns_batch_{batch_number + 1}.parquet"
        merged_fgs1_signal_df.to_parquet(output_path)
        print(f"Saved merged FGS1 signal columns for batch to {output_path}, shape: {merged_fgs1_signal_df.shape}")


















