import os
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from dataloaders.datasets.AgeDB import AgeDB
from dataloaders.datasets.AgeDB_Unlabeled import AgeDB_Unlabeled
from dataloaders.datasets.UTKFace import UTKFace
from dataloaders.datasets.UTKFace_Unlabeled import UTKFace_Unlabeled
from dataloaders.datasets.So2Sat_POP import So2Sat_POP
from dataloaders.datasets.So2Sat_POP_Unlabeled import So2Sat_POP_Unlabeled
from dataloaders.datasets.Bayern_ForestHeight import Bayern_ForestHeight
from dataloaders.datasets.Bayern_ForestHeight_Unlabeled import Bayern_ForestHeight_Unlabeled


def compute_dem_stats(data_dir, df, num_samples=1000):
    """
    Compute DEM min/max statistics from a sample of images.
    
    Args:
        data_dir: Base data directory
        df: DataFrame with image paths
        num_samples: Number of images to sample for statistics
    
    Returns:
        (dem_min, dem_max) tuple
    """
    import random
    
    # Sample a subset of images for efficiency
    sample_size = min(num_samples, len(df))
    sample_paths = random.sample(list(df['path'].values), sample_size)
    
    all_mins, all_maxs = [], []
    for path in sample_paths:
        img_path = os.path.join(data_dir, 'So2Sat_POP', path)
        try:
            img = Image.open(img_path)
            img_array = np.array(img, dtype=np.float32)
            all_mins.append(img_array.min())
            all_maxs.append(img_array.max())
        except Exception:
            continue
    
    if len(all_mins) == 0:
        # Fallback to default values
        return -2.0, 2.0
    
    # Use percentiles to be robust to outliers
    dem_min = float(np.percentile(all_mins, 1))  # 1st percentile
    dem_max = float(np.percentile(all_maxs, 99))  # 99th percentile
    
    # Add small margin
    margin = (dem_max - dem_min) * 0.05
    dem_min -= margin
    dem_max += margin
    
    print(f"DEM statistics computed from {len(all_mins)} images: min={dem_min:.4f}, max={dem_max:.4f}")
    return dem_min, dem_max


def compute_sen2_stats(data_dir, df, num_samples=200, seed=42):
    """
    Compute global min/max statistics for Sentinel-2 RGB bands after
    clipping to [0, 4000] and scaling to [0, 1].
    """
    try:
        import tifffile
    except ImportError:
        print("tifffile not available; using default sen2 min/max of [0,1].")
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    import random
    random.seed(seed)

    sample_size = min(num_samples, len(df))
    sample_paths = random.sample(list(df['path'].values), sample_size)

    band_mins = [np.inf, np.inf, np.inf]
    band_maxs = [-np.inf, -np.inf, -np.inf]

    for path in sample_paths:
        img_path = os.path.join(data_dir, 'So2Sat_POP', path)
        try:
            img_array = tifffile.imread(img_path)
            image_bands = img_array[:, :, [3, 2, 1]].astype(np.float32)
            image_bands = np.clip(image_bands, 0, 4000) / 4000.0

            for i in range(3):
                band = image_bands[:, :, i]
                band_mins[i] = min(band_mins[i], float(band.min()))
                band_maxs[i] = max(band_maxs[i], float(band.max()))
        except Exception:
            continue

    if any(np.isinf(v) for v in band_mins) or any(np.isinf(v) for v in band_maxs):
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    print(
        "Sentinel-2 stats (post-clip/scale): "
        f"min={band_mins}, max={band_maxs} (from {sample_size} samples)"
    )
    return band_mins, band_maxs


def make_semi_loader(args, num_workers=12):
    # Handle CSV filename - so2sat_pop uses simreg_ prefix and data source suffix
    if args.dataset.lower() == 'so2sat_pop':
        data_source = getattr(args, 'data_source', 'sen2')  # default to sen2
        if data_source == 'sen2':
            csv_filename = 'simreg_so2sat_pop_sen2.csv'
            print(f"Using Sentinel-2 satellite imagery (RGB bands)")
        else:
            csv_filename = 'simreg_so2sat_pop.csv'
            print(f"Using DEM (elevation) data")
    elif args.dataset.lower() in ['bayern_forest', 'simreg_bayern_forest']:
        csv_filename = 'simreg_bayern_forest.csv'
    else:
        csv_filename = f'{args.dataset}.csv'
    
    df = pd.read_csv(os.path.join(args.data_dir, csv_filename))
    
    # Apply log-transform to labels if enabled (before any splits)
    if args.log_transform:
        original_range = (df['age'].min(), df['age'].max())
        df['age'] = np.log1p(df['age'])  # log(1 + x) to handle zeros
        transformed_range = (df['age'].min(), df['age'].max())
        print(f"Log-transform applied: range [{original_range[0]:.2f}, {original_range[1]:.2f}] -> [{transformed_range[0]:.4f}, {transformed_range[1]:.4f}]")
    
    df_train, df_val, df_test = df[df['split'] == 'train'].copy(), df[df['split'] == 'val'].copy(), df[df['split'] == 'test'].copy()
    
    df_train = make_balanced_unlabeled(df_train, args)
    df_labeled, df_unlabeled = df_train[df_train['split_train']=='labeled'], df_train[df_train['split_train']=='unlabeled']
    df_labeled = make_reduced(df_labeled, args)
    df_labeled = df_labeled[df_labeled['split_train_reduced']=='use']

    # Compute label normalization statistics from training data only (if enabled)
    label_mean = None
    label_std = None
    if args.normalize_labels:
        # Use fixed normalization values for so2sat_pop dataset
        if args.dataset.lower() == 'so2sat_pop':
            label_mean = 1085.0
            label_std = 2800.0
            print(f"Using fixed normalization for so2sat_pop: mean={label_mean}, std={label_std}")
        else:
            label_mean = float(df_labeled['age'].mean())
            label_std = float(df_labeled['age'].std())
            # Avoid division by zero
            if label_std < 1e-6:
                label_std = 1.0
            print(f"Label normalization enabled: mean={label_mean:.2f}, std={label_std:.2f}")
        print(f"Label range: [{df_labeled['age'].min():.2f}, {df_labeled['age'].max():.2f}]")
    else:
        print("Label normalization disabled (using raw labels)")

    # Store normalization stats in args for later use
    args.label_mean = label_mean
    args.label_std = label_std

    # Select dataset class based on dataset name
    if args.dataset.lower() == 'utkface':
        LabeledDataset = UTKFace
        UnlabeledDataset = UTKFace_Unlabeled
        dem_min, dem_max = None, None  # Not used for UTKFace
    elif args.dataset.lower() == 'so2sat_pop':
        LabeledDataset = So2Sat_POP
        UnlabeledDataset = So2Sat_POP_Unlabeled
        # Compute DEM statistics only for DEM data (not needed for Sentinel-2)
        data_source = getattr(args, 'data_source', 'sen2')
        if data_source == 'dem':
            dem_min, dem_max = compute_dem_stats(args.data_dir, df_train)
            sen2_min, sen2_max = None, None
        else:
            dem_min, dem_max = None, None  # Not needed for Sentinel-2
            sen2_min, sen2_max = compute_sen2_stats(args.data_dir, df_train, seed=args.seed)
    elif args.dataset.lower() in ['bayern_forest', 'simreg_bayern_forest']:
        LabeledDataset = Bayern_ForestHeight
        UnlabeledDataset = Bayern_ForestHeight_Unlabeled
        dem_min, dem_max = None, None  # Not used for Bayern Forest (normalization done in dataset)
    else:
        LabeledDataset = AgeDB
        UnlabeledDataset = AgeDB_Unlabeled
        dem_min, dem_max = None, None  # Not used for AgeDB

    # Create dataset kwargs (only pass dem stats for So2Sat_POP with DEM data)
    if args.dataset.lower() == 'so2sat_pop':
        dataset_kwargs = {
            'dem_min': dem_min,
            'dem_max': dem_max,
            'sen2_min': sen2_min,
            'sen2_max': sen2_max,
        }
    else:
        dataset_kwargs = {}

    labeled_set = LabeledDataset(data_dir=args.data_dir,
                        df = df_labeled,
                        img_size=args.img_size,
                        split='train',
                        label_mean=label_mean,
                        label_std=label_std,
                        **dataset_kwargs
                        )
    labeled_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    unlabeled_set = UnlabeledDataset(data_dir=args.data_dir,
                        df = df_unlabeled,
                        img_size=args.img_size,
                        split='train',
                        label_mean=label_mean,
                        label_std=label_std,
                        **dataset_kwargs
                        )
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    valid_set = LabeledDataset(data_dir=args.data_dir,
                        df = df_val,
                        img_size=args.img_size,
                        split='valid',
                        label_mean=label_mean,
                        label_std=label_std,
                        **dataset_kwargs
                        )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        
    test_set = LabeledDataset(data_dir=args.data_dir,
                        df = df_test,
                        img_size=args.img_size,
                        split='test',
                        label_mean=label_mean,
                        label_std=label_std,
                        **dataset_kwargs
                        )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return labeled_loader, unlabeled_loader, valid_loader, test_loader


def make_balanced_unlabeled(data, args):
    import random
    random.seed(args.seed)
    
    # Get actual age range from data instead of hardcoded 0-100
    min_age = int(data['age'].min())
    max_age = int(data['age'].max())
    unique_values = data['age'].nunique()
    
    # For high-cardinality targets (like population data), use stratified binning
    # Threshold: if more than 1000 unique values, use stratified binning
    HIGH_CARDINALITY_THRESHOLD = 1000
    NUM_BINS = 100
    
    l_set, u_set = [], []
    
    if unique_values > HIGH_CARDINALITY_THRESHOLD:
        # Simple random split for high-cardinality targets
        print(f"High-cardinality target detected ({unique_values} unique values). Using simple random 50/50 split.")
        
        all_paths = list(data['path'].values)
        random.shuffle(all_paths)
        split_point = len(all_paths) // 2
        l_set = all_paths[:split_point]
        u_set = all_paths[split_point:]
    else:
        # Original exact-value matching strategy (for age datasets like UTKFace)
        age_range = range(min_age, max_age + 1)
        for v in age_range:
            curr_df = data[data['age']==v]
            curr_data = list(curr_df['path'].values)
            random.shuffle(curr_data)
            
            curr_size = len(curr_data) // 2
            l_set += curr_data[:curr_size]
            u_set += curr_data[curr_size:]
    
    print(f"Labeled Data: {len(l_set)} | Unlabeled Data: {len(u_set)}")
    
    assert len(set(l_set).intersection(set(u_set)))==0
    
    combined_set = dict(zip(l_set, ['labeled' for _ in range(len(l_set))]))
    combined_set.update(dict(zip(u_set, ['unlabeled' for _ in range(len(u_set))])))

    data['split_train'] = data['path'].map(combined_set)
    return data


def make_reduced(data, args):
    import random
    random.seed(args.seed)
    
    # Get actual age range from data instead of hardcoded 0-100
    min_age = int(data['age'].min())
    max_age = int(data['age'].max())
    unique_values = data['age'].nunique()
    
    # For high-cardinality targets, use stratified binning strategy
    HIGH_CARDINALITY_THRESHOLD = 1000
    NUM_BINS = 100
    
    use_set, not_set = [], []
    
    if unique_values > HIGH_CARDINALITY_THRESHOLD:
        # STRATIFIED: Use log-spaced bins for skewed data
        print(f"Using stratified reduction with labeled_ratio={args.labeled_ratio}")
        
        # Separate zero values (common in population data)
        zero_mask = data['age'] == 0
        zero_data = data[zero_mask]
        nonzero_data = data[~zero_mask]
        
        # Handle zero-valued samples separately
        if len(zero_data) > 0:
            zero_paths = list(zero_data['path'].values)
            random.shuffle(zero_paths)
            curr_size = max(1, int(len(zero_paths) * args.labeled_ratio))
            use_set += zero_paths[:curr_size]
            not_set += zero_paths[curr_size:]
            print(f"  Zero-value: using {curr_size}/{len(zero_paths)}")
        
        # For non-zero values, use log-spaced bins for better coverage
        if len(nonzero_data) > 0:
            log_values = np.log1p(nonzero_data['age'].values)
            
            # Create bins based on log-transformed values
            percentiles = np.linspace(0, 100, NUM_BINS + 1)
            log_bins = np.percentile(log_values, percentiles)
            log_bins = np.unique(log_bins)
            
            if len(log_bins) >= 2:
                nonzero_data = nonzero_data.copy()
                nonzero_data['log_age'] = log_values
                nonzero_data['age_bin'] = pd.cut(nonzero_data['log_age'], bins=log_bins,
                                                  include_lowest=True, duplicates='drop')
                
                # Split within each bin according to labeled_ratio
                for bin_label in nonzero_data['age_bin'].dropna().unique():
                    curr_df = nonzero_data[nonzero_data['age_bin'] == bin_label]
                    curr_data = list(curr_df['path'].values)
                    random.shuffle(curr_data)
                    
                    curr_size = max(1, int(len(curr_data) * args.labeled_ratio))
                    use_set += curr_data[:curr_size]
                    not_set += curr_data[curr_size:]
                
                # Handle any samples that didn't get binned
                binned_paths = set(use_set + not_set)
                unbinned = nonzero_data[~nonzero_data['path'].isin(binned_paths)]
                if len(unbinned) > 0:
                    unbinned_paths = list(unbinned['path'].values)
                    random.shuffle(unbinned_paths)
                    split_point = max(1, int(len(unbinned_paths) * args.labeled_ratio))
                    use_set += unbinned_paths[:split_point]
                    not_set += unbinned_paths[split_point:]
            else:
                # Fallback: simple random split
                nonzero_paths = list(nonzero_data['path'].values)
                random.shuffle(nonzero_paths)
                split_point = max(1, int(len(nonzero_paths) * args.labeled_ratio))
                use_set += nonzero_paths[:split_point]
                not_set += nonzero_paths[split_point:]
            
            nonzero_used = len([p for p in use_set if p in set(nonzero_data['path'].values)])
            print(f"  Non-zero: using {nonzero_used}/{len(nonzero_data)}")
    else:
        # Original exact-value matching strategy (for age datasets like UTKFace)
        age_range = range(min_age, max_age + 1)
        for v in age_range:
            curr_df = data[data['age']==v]
            curr_data = list(curr_df['path'].values)
            random.shuffle(curr_data)
            
            curr_size = int(len(curr_data) * args.labeled_ratio)
            use_set += curr_data[:curr_size]
            not_set += curr_data[curr_size:]
    
    print(f"Using Data: {len(use_set)} | Not using Data: {len(not_set)}")
    
    assert len(set(use_set).intersection(set(not_set)))==0
    
    combined_set = dict(zip(use_set, ['use' for _ in range(len(use_set))]))
    combined_set.update(dict(zip(not_set, ['not' for _ in range(len(not_set))])))

    data['split_train_reduced'] = data['path'].map(combined_set)

    return data
