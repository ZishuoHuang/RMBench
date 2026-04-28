import h5py

def print_hdf5_structure(name, obj):
    print(f"{name}: {type(obj)}")
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")

with h5py.File('/home/CNS2026497693/RoboTwin/data/pick_diverse_bottles/aloha-agilex_clean_50/data_done/episode0.hdf5', 'r') as f:
    f.visititems(print_hdf5_structure)
