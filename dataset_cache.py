import torch
from torch.utils.data import Dataset, DataLoader
import grand.dataio.root_files as froot
import glob
from collections import OrderedDict

class GRAND_DC2_TracesDataset(Dataset):
    def __init__(self, rootpath='/sps/grand/DC2Training/ZHAireS', dataset='NJ', what='efield', level='L0', mode='event', cache_size=2):
        self.datadir = f"{rootpath}-{dataset}"
        self.what = what
        self.level = level
        self.mode = mode
        self.event_count = 2  # 12 files x 1000 events
        self.cache_size = cache_size
        self.event_cache = OrderedDict()  
        if mode == 'trace':
            self._compute_trace_indices()

    def _compute_trace_indices(self):
        self.trace_indices = []
        for event_idx in range(self.event_count):
            filename = self._get_file_name(event_idx)
            ef3d = froot.get_handling3dtraces(filename, event_idx % 1000)
            n_trig_antennas = ef3d.traces.shape[0]
            self.trace_indices.extend([(event_idx, i) for i in range(n_trig_antennas)])

    def __len__(self):
        if self.mode == 'event':
            return self.event_count
        elif self.mode == 'trace':
            return len(self.trace_indices)

    def __getitem__(self, idx):
        if self.mode == 'event':
            return self._get_event(idx)
        elif self.mode == 'trace':
            event_idx, ant_idx = self.trace_indices[idx]
            return self._get_trace(event_idx, ant_idx)

    def _get_event(self, idx):
        filename = self._get_file_name(idx)
        ef3d = froot.get_handling3dtraces(filename, idx % 1000)
        traces = ef3d.traces  
        triggered_antenna_indices = ef3d.idx2idt  
        return traces, triggered_antenna_indices
    
    def _get_trace(self, event_idx, ant_idx):
        if event_idx not in self.event_cache:
            if len(self.event_cache) >= self.cache_size:
                self.event_cache.popitem(last=False)  
            filename = self._get_file_name(event_idx)
            self.event_cache[event_idx] = froot.get_handling3dtraces(filename, event_idx % 1000)
        trace = self.event_cache[event_idx].traces[ant_idx]  
        return trace
    
    def _get_file_name(self, index):
        num_file = index // 1000
        fileroot = 'sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_'
        filename = f"{fileroot}{str(num_file).rjust(4, '0')}"
        efield_file = glob.glob(f"{self.datadir}/{filename}/{self.what}*{self.level}*")
        if not efield_file:
            raise FileNotFoundError(f"File not found for index {index}")
        return efield_file[0]

def collate_fn_event(batch):
    batch_traces = [torch.as_tensor(traces) for traces, _ in batch]  
    batch_indices = [indices for _, indices in batch]
    return batch_traces, batch_indices

dataset = GRAND_DC2_TracesDataset(mode='trace')
collate_fn = collate_fn_event if dataset.mode == 'event' else None
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, drop_last=True)

for i, batch in enumerate(dataloader):
    if dataset.mode == 'event':
        batch_traces, batch_indices = batch 
        print(f"Batch {i}: {len(batch_traces)} événements")  
        for j, traces in enumerate(batch_traces):
            print(f" - Event {j}: {traces.shape}") 
    elif dataset.mode == 'trace':
        traces = batch
        print(batch.shape)
    else:
        raise NotImplementedError("Mode can be trace or event")
