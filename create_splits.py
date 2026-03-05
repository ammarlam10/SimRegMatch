"""
Script to create train/val/test splits for So2Sat_POP (Sentinel-2 spring only).

Rules:
- Train: first 90% from So2Sat_POP_Part1/train
- Val: last 10% from So2Sat_POP_Part1/train
- Test: all from So2Sat_POP_Part1/test
- Use only sen2spring imagery
- Downsample labels <= 100 to keep 20% (train only)
"""

import os
import numpy as np
import pandas as pd

# Configuration
RANDOM_SEED = 42
# Use environment variable if set (for Docker), otherwise use default path
DATA_DIR = os.getenv("SO2SAT_DATA_DIR", "/work/ammar/sslrp/data/So2Sat_POP")
INPUT_CSV = os.path.join(DATA_DIR, "FileList.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "simreg_so2sat_pop_sen2.csv")

# Split ratios
TRAIN_RATIO = 0.9
LOW_LABEL_THRESHOLD = 100
LOW_LABEL_KEEP_RATIO = 0.2

# Paths and filters
TRAIN_PREFIX = "So2Sat_POP_Part1/train/"
TEST_PREFIX = "So2Sat_POP_Part1/test/"
SEN2_SEASON = "sen2spring"

def main():
    rng = np.random.RandomState(RANDOM_SEED)

    print(f"Reading CSV from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Normalize column names
    if "FileName" in df.columns:
        df = df.rename(columns={"FileName": "path"})
    if "POP" in df.columns:
        df = df.rename(columns={"POP": "age"})
    
    # Convert paths to sen2spring (CSV has sen2summer, but we want sen2spring)
    # Replace both directory path and filename
    df["path"] = df["path"].str.replace(r"/sen2(autumn|summer|winter|spring)/", f"/{SEN2_SEASON}/", regex=True)
    df["path"] = df["path"].str.replace(r"_sen2(autumn|summer|winter|spring)\.tif", f"_{SEN2_SEASON}.tif", regex=True)
    
    # Filter to only paths that contain sen2spring
    df = df[df["path"].str.contains(f"/{SEN2_SEASON}/", regex=False)]

    # Split by path prefix (deterministic order as in CSV)
    df_train_all = df[df["path"].str.startswith(TRAIN_PREFIX)].copy().reset_index(drop=True)
    df_test = df[df["path"].str.startswith(TEST_PREFIX)].copy().reset_index(drop=True)

    n_total = len(df_train_all)
    n_train = int(n_total * TRAIN_RATIO)

    df_train = df_train_all.iloc[:n_train].copy()
    df_val = df_train_all.iloc[n_train:].copy()

    print(f"Train (pre-reduction): {len(df_train)}")
    print(f"Val: {len(df_val)}")
    print(f"Test: {len(df_test)}")
    
    # Downsample low labels in train only
    low_mask = df_train["age"] <= LOW_LABEL_THRESHOLD
    low_df = df_train[low_mask]
    if len(low_df) > 0:
        keep_count = int(len(low_df) * LOW_LABEL_KEEP_RATIO)
        keep_count = max(1, keep_count)
        keep_idx = rng.choice(low_df.index.values, size=keep_count, replace=False)
        df_train = pd.concat([df_train[~low_mask], low_df.loc[keep_idx]]).sort_index()
        print(f"Reduced low-label train samples: {len(low_df)} -> {keep_count}")
    else:
        print("No low-label samples (<=100) found in train split.")
    
    # Assign split labels
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    # Keep required columns
    df_out = pd.concat([df_train, df_val, df_test], axis=0)
    df_out = df_out[["path", "age", "split"]]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCSV saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
