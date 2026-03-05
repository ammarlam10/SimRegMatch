import h5py
import numpy as np

# Open one of the HDF5 files to inspect its structure
with h5py.File('/workspace/external_data/Bayern_forest_height_reduced/32744_5288_20.h5', 'r') as f:
    print('Keys in HDF5 file:', list(f.keys()))

    for key in f.keys():
        data = f[key]
        print(f'\nDataset: {key}')
        print(f'Shape: {data.shape}')
        print(f'Dtype: {data.dtype}')
        if hasattr(data, 'attrs') and data.attrs:
            print(f'Attributes: {dict(data.attrs)}')

        # Show some basic statistics if it's a numerical array
        if data.dtype.kind in 'iufc':  # integer, unsigned, float, complex
            print(f'Min: {np.min(data)}')
            print(f'Max: {np.max(data)}')
            print(f'Mean: {np.mean(data)}')
            print(f'Std: {np.std(data)}')
            print(f'NaN count: {np.sum(np.isnan(data))}')
            print(f'Zero count: {np.sum(data == 0)}')