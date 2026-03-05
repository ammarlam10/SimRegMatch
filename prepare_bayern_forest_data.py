"""
Prepare Bayern Forest Height dataset for SimRegMatch training.
Converts splits.csv + ndsm_stats.csv to SimRegMatch format.
"""
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

def prepare_bayern_forest_data(data_dir, output_dir):
    """
    Convert Bayern Forest Height dataset to SimRegMatch format.
    
    Args:
        data_dir: Path to Bayern_forest_height_reduced directory
        output_dir: Path to output directory for CSV file
    """
    print("=" * 60)
    print("Preparing Bayern Forest Height Dataset for SimRegMatch")
    print("=" * 60)
    
    # Read splits.csv
    splits_path = os.path.join(data_dir, 'splits.csv')
    print(f"\nReading splits from: {splits_path}")
    df_splits = pd.read_csv(splits_path)
    
    # Read ndsm_stats.csv
    stats_path = os.path.join(data_dir, 'ndsm_stats.csv')
    print(f"Reading statistics from: {stats_path}")
    df_stats = pd.read_csv(stats_path)
    
    # Create a dictionary for quick lookup of mean height per file
    file_mean_dict = dict(zip(df_stats['filename'], df_stats['mean']))
    
    print(f"\nTotal patches: {len(df_splits)}")
    print(f"Unique files: {df_splits['filename'].nunique()}")
    print(f"Split distribution:")
    print(df_splits['split'].value_counts())
    
    # Compute per-patch mean height from HDF5 files
    print("\nComputing per-patch statistics from HDF5 files...")
    patch_means = []
    
    for idx, row in tqdm(df_splits.iterrows(), total=len(df_splits), desc="Processing patches"):
        filename = row['filename']
        patch_idx = row['patch_idx']
        
        # Load HDF5 file and get patch mean
        h5_path = os.path.join(data_dir, filename)
        try:
            with h5py.File(h5_path, 'r') as f:
                ndsm = f['ndsm'][patch_idx]  # Shape: (256, 256, 1)
                patch_mean = float(np.mean(ndsm))
                patch_means.append(patch_mean)
        except Exception as e:
            print(f"\nWarning: Could not read {filename} patch {patch_idx}: {e}")
            # Fallback to file-level mean
            patch_means.append(file_mean_dict.get(filename, 12.13))
    
    # Create SimRegMatch format DataFrame
    df_simreg = pd.DataFrame({
        'path': df_splits['filename'] + ',' + df_splits['patch_idx'].astype(str),
        'age': patch_means,  # Use 'age' column name for compatibility
        'split': df_splits['split']
    })
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'simreg_bayern_forest.csv')
    df_simreg.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 60}")
    print(f"Dataset prepared successfully!")
    print(f"{'=' * 60}")
    print(f"Output file: {output_path}")
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(df_simreg)}")
    print(f"  Train: {len(df_simreg[df_simreg['split'] == 'train'])} ({len(df_simreg[df_simreg['split'] == 'train'])/len(df_simreg)*100:.1f}%)")
    print(f"  Val: {len(df_simreg[df_simreg['split'] == 'val'])} ({len(df_simreg[df_simreg['split'] == 'val'])/len(df_simreg)*100:.1f}%)")
    print(f"  Test: {len(df_simreg[df_simreg['split'] == 'test'])} ({len(df_simreg[df_simreg['split'] == 'test'])/len(df_simreg)*100:.1f}%)")
    print(f"\nHeight statistics (meters):")
    print(f"  Mean: {df_simreg['age'].mean():.2f}")
    print(f"  Std: {df_simreg['age'].std():.2f}")
    print(f"  Min: {df_simreg['age'].min():.2f}")
    print(f"  Max: {df_simreg['age'].max():.2f}")
    print(f"{'=' * 60}\n")
    
    return df_simreg


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Bayern Forest Height dataset')
    parser.add_argument('--data-dir', type=str, 
                        default='/home/ammar/data/Bayern_forest_height_reduced',
                        help='Path to Bayern_forest_height_reduced directory')
    parser.add_argument('--output-dir', type=str,
                        default='/home/ammar/data',
                        help='Output directory for CSV file')
    
    args = parser.parse_args()
    
    prepare_bayern_forest_data(args.data_dir, args.output_dir)
