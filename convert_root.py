import h5py
import numpy as np
import os
import time
import glob
import uproot
import awkward as ak
from tqdm import tqdm

def convert_root_to_h5(root_dir, h5_filename, level='L0', what='efield', max_files=1, max_events=1):
    print(f"Starting conversion of {root_dir} to {h5_filename}...")
    start_time = time.time()
    
    file_pattern = f"{root_dir}/*/{what}*{level}*"
    root_files = glob.glob(file_pattern)[:max_files]
    if not root_files:
        raise FileNotFoundError(f"No {level} files found in {root_dir}")
    
    event_metadata = []
    total_traces = 0
    
    for file_path in tqdm(root_files, desc="Scanning files"):
        with uproot.open(file_path) as f:
            print(f"Opening file: {file_path}")
            
            n_events_total = f["tefield"].num_entries
            print(f"File contains {n_events_total} events")
            
            n_to_process = min(n_events_total, max_events)
            
            for evt_idx in range(n_to_process):
                event_array = f["tefield"]["trace"].array(entry_start=evt_idx, entry_stop=evt_idx+1)[0]
                nb_tr = len(event_array)
                event_metadata.append((total_traces, nb_tr))
                total_traces += nb_tr
                print(f"Event {evt_idx}: {nb_tr} antennas, total traces: {total_traces}")
    
    with h5py.File(h5_filename, 'w') as h5f:
        print(f"Creating HDF5 file with {total_traces} total traces")
        chunk_size = min(64, total_traces)
        
        traces_ds = h5f.create_dataset(
            'traces',
            shape=(total_traces, 3, 8192),
            dtype=np.float32,
            chunks=(chunk_size, 3, 8192)
        )
        
        meta_ds = h5f.create_dataset(
            'event_metadata',
            data=np.array(event_metadata, dtype=np.uint32)
        )
        
        trace_idx = 0
        for file_path in tqdm(root_files, desc="Processing files"):
            with uproot.open(file_path) as f:
                n_events_total = f["tefield"].num_entries
                n_to_process = min(n_events_total, max_events)
                
                for evt_idx in range(n_to_process):
                    event_array = f["tefield"]["trace"].array(entry_start=evt_idx, entry_stop=evt_idx+1)[0]
                    tr3d = ak.to_numpy(event_array)
                    nb_tr = tr3d.shape[0]
                    print(f"Converting event {evt_idx}, shape: {tr3d.shape}")
                    traces_ds[trace_idx:trace_idx+nb_tr] = tr3d
                    trace_idx += nb_tr

    total_time = time.time() - start_time
    print(f"Conversion completed in {total_time:.2f}s")
    print(f"Total events: {len(event_metadata)}")
    print(f"Total traces: {total_traces}")
    return h5_filename
    
if __name__=='__main__':
    convert_root_to_h5(
        root_dir="/sps/grand/DC2Training/ZHAireS-NJ",
        h5_filename="debug_dataset.h5",
        max_files=1,
        max_events=1
    )
