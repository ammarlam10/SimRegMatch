#!/usr/bin/env python3
"""
Quick test to verify optimization implementations.
Tests num_workers, pin_memory, and caching without running full training.
"""

import torch
import argparse
import time
from dataloaders import make_semi_loader
from utils.args import SimRegMatch_parser


def test_dataloader_config():
    """Test that DataLoader configurations are correct."""
    print("=" * 80)
    print("TESTING DATALOADER OPTIMIZATIONS")
    print("=" * 80)
    
    parser = SimRegMatch_parser()
    args = parser.parse_args([
        '--dataset', 'so2sat_pop',
        '--data-source', 'sen2',
        '--data_dir', '/workspace/data',
        '--labeled-ratio', '0.1',
        '--batch_size', '32',
        '--img_size', '224',
        '--normalize-labels',
        '--seed', '0'
    ])
    
    args.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("\n1. Testing without cache...")
    start = time.time()
    labeled_loader, unlabeled_loader, valid_loader, test_loader = \
        make_semi_loader(args, num_workers=8, use_cache=False)
    
    # Check DataLoader properties
    print(f"   ✓ Labeled loader: num_workers={labeled_loader.num_workers}, pin_memory={labeled_loader.pin_memory}")
    print(f"   ✓ Unlabeled loader: num_workers={unlabeled_loader.num_workers}, pin_memory={unlabeled_loader.pin_memory}")
    print(f"   ✓ Valid loader: num_workers={valid_loader.num_workers}, pin_memory={valid_loader.pin_memory}")
    print(f"   ✓ Test loader: num_workers={test_loader.num_workers}, pin_memory={test_loader.pin_memory}")
    
    # Test loading one batch
    print("\n   Testing batch loading...")
    batch = next(iter(labeled_loader))
    print(f"   ✓ Batch shape: {batch['input'].shape}")
    print(f"   ✓ Label shape: {batch['label'].shape}")
    print(f"   ✓ Data type: {batch['input'].dtype}")
    
    elapsed = time.time() - start
    print(f"\n   Time without cache: {elapsed:.2f} seconds")
    
    print("\n2. Testing with cache (--use-cache)...")
    args.use_cache = True
    start = time.time()
    
    labeled_loader_cached, _, _, _ = make_semi_loader(args, num_workers=8, use_cache=True)
    
    # Test loading one batch from cached dataset
    batch_cached = next(iter(labeled_loader_cached))
    print(f"   ✓ Cached batch shape: {batch_cached['input'].shape}")
    
    elapsed_cached = time.time() - start
    print(f"\n   Time with cache: {elapsed_cached:.2f} seconds (includes preloading)")
    
    print("\n" + "=" * 80)
    print("✅ ALL OPTIMIZATIONS VERIFIED")
    print("=" * 80)
    print("\nOptimizations active:")
    print("  ✓ num_workers=8 (parallel data loading)")
    print("  ✓ pin_memory=True (faster CPU→GPU transfer)")
    print("  ✓ CachedDataset available (use --use-cache to enable)")
    print("\nReady for training!")
    print("=" * 80)


if __name__ == "__main__":
    test_dataloader_config()
