#!/usr/bin/env python3
"""
Script to prepare UTKFace CSV file for SimRegMatch
Converts FileList.csv format to the required format
"""
import pandas as pd
import os
import sys

def prepare_utkface_csv(input_csv, output_csv):
    """
    Convert UTKFace CSV to SimRegMatch format
    
    Args:
        input_csv: Path to original FileList.csv
        output_csv: Path to output utkface.csv
    """
    print(f"Reading CSV from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Rename FileName to path
    if 'FileName' in df.columns:
        df = df.rename(columns={'FileName': 'path'})
    
    # Convert SPLIT to lowercase split
    if 'SPLIT' in df.columns:
        df['split'] = df['SPLIT'].str.lower()
        df = df.drop(columns=['SPLIT'])
    
    # Keep only required columns: path, age, split
    required_cols = ['path', 'age', 'split']
    df = df[required_cols]
    
    # Verify split values
    print(f"\nSplit distribution:")
    print(df['split'].value_counts())
    
    # Verify age range
    print(f"\nAge range: {df['age'].min()} - {df['age'].max()}")
    
    # Save to output
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nProcessed CSV saved to: {output_csv}")
    print(f"Final shape: {df.shape}")
    print(f"Final columns: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    # Default paths (using container paths)
    input_csv = '/workspace/ucvme/DATA_DIR/FileList.csv'
    output_csv = '/workspace/data/utkface.csv'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    prepare_utkface_csv(input_csv, output_csv)
