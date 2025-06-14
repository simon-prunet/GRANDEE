import os
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split
import grand.dataio.root_files as froot
import glob
import numpy as np
from functools import lru_cache
import time
from torch.profiler import profile, record_function, ProfilerActivity
import logging
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import re 

import basicblock as B

import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GRAND_H5_TracesDataset(Dataset):
    def __init__(self, h5_file_path='/sps/grand/selbouch/h5_files/grand_data.h5', 
                 num_realizations=5, snr_range_db=(-10, 20), normalize=True, 
                 chunk_size=200, cache_size=2):
        
        self.h5_file_path = h5_file_path
        self.num_realizations = num_realizations
        self.snr_range_db = snr_range_db  # SNR range in dB
        self.normalize = normalize
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        
        with h5py.File(h5_file_path, 'r') as f:
            self.total_traces = f['traces'].shape[0]
            self.trace_shape = f['traces'].shape[1:]
        
        self._length = self.total_traces * num_realizations
        self.cache = {}
        self.cache_order = []
        
        print(f"Dataset: {self.total_traces} traces of shape {self.trace_shape}")
        print(f"SNR range: {snr_range_db[0]} to {snr_range_db[1]} dB")
    
    def _get_chunk_id(self, trace_idx):
        return trace_idx // self.chunk_size

    def _load_chunk(self, chunk_id):
        if chunk_id in self.cache:
            # Move to end for LRU
            self.cache_order.remove(chunk_id)
            self.cache_order.append(chunk_id)
            return self.cache[chunk_id]
        
        # Load new chunk
        start_idx = chunk_id * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_traces)
        
        with h5py.File(self.h5_file_path, 'r') as f:
            chunk_data = torch.from_numpy(f['traces'][start_idx:end_idx]).float()
        
        # Cache management
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[chunk_id] = chunk_data
        self.cache_order.append(chunk_id)
        
        return chunk_data

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        trace_idx = idx // self.num_realizations
        chunk_id = self._get_chunk_id(trace_idx)
        
        chunk_data = self._load_chunk(chunk_id)
        chunk_offset = trace_idx - (chunk_id * self.chunk_size)
        clean_trace = chunk_data[chunk_offset].clone()
        
        # Normalize if requested
        if self.normalize:
            # Use robust normalization
            global_std = torch.std(clean_trace) + 1e-8
            global_mean = torch.mean(clean_trace)
            clean_trace = (clean_trace - global_mean) / global_std
        
        # SNR-based noise addition
        snr_db = torch.empty(1).uniform_(*self.snr_range_db).item()
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate signal power (per channel)
        signal_power = torch.mean(clean_trace ** 2, dim=-1, keepdim=True)
        
        # Calculate noise power based on desired SNR
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(noise_power)
        
        # Generate noise with appropriate power
        noise = torch.randn_like(clean_trace) * noise_std
        noisy_trace = clean_trace + noise
        
        # Convert SNR to a normalized sigma value for the model
        # Map SNR range to a reasonable sigma range (e.g., 0.01 to 1.0)
        sigma_min, sigma_max = 0.01, 1.0
        snr_min, snr_max = self.snr_range_db
        sigma = sigma_min + (sigma_max - sigma_min) * (snr_max - snr_db) / (snr_max - snr_min)
        
        return (
            torch.tensor(sigma, dtype=torch.float32),
            noisy_trace,
            clean_trace
        )

    def close(self):
        """Clean up cache"""
        if hasattr(self, 'cache'):
            self.cache.clear()
            self.cache_order.clear()
        
    def __del__(self):
        """Clean up cache when object is deleted"""
        if hasattr(self, 'cache'):
            self.cache.clear()
            self.cache_order.clear()

class UNetRes(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, n_downs=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(UNetRes, self).__init__()

        self.n_downs = n_downs 
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        for nd in range(self.n_downs):
            setattr(self, 'm_down%d'%(nd+1), B.sequential(*[B.ResBlock(nc[nd], nc[nd], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[nd], nc[nd+1], bias=False, mode='2')))
       
        self.m_body  = B.sequential(*[B.ResBlock(nc[self.n_downs], nc[self.n_downs], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        for nu in range(self.n_downs,0,-1):
            setattr(self, 'm_up%d'%(nu), B.sequential(upsample_block(nc[nu], nc[nu-1], bias=False, mode='2'), *[B.ResBlock(nc[nu-1], nc[nu-1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)]))
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x0):
        xs = []
        xs.append(self.m_head(x0))
        for nd in range(self.n_downs):
            xs.append(getattr(self,'m_down%d'%(nd+1))(xs[-1]))
        x = self.m_body(xs[-1])

        for i, nu in enumerate(range(self.n_downs,0,-1)):
            x = getattr(self, 'm_up%d'%nu)(x+xs[-1-i])

        x = self.m_tail(x+xs[0])

        return x

class PRUNet(UNetRes):
    def __init__(self, in_nc=3, n_downs=3, nc=[64, 128, 256, 512], nb=2, use_batch_norm=True):
        act_mode = 'R'
        mode_str = 'CB' + act_mode if use_batch_norm else 'C' + act_mode
        
        super(PRUNet, self).__init__(
            in_nc=in_nc+1, 
            out_nc=in_nc, 
            n_downs=n_downs, 
            nc=nc, 
            nb=nb,
            act_mode=act_mode
        )
    
    def forward(self, x, sigma=None):
        if sigma is None:
            sigma = torch.ones(1, device=x.device) * 0.1
            
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=x.device, dtype=torch.float)
            
        if sigma.numel() == 1:  
            sigma = sigma.expand(x.shape[0])
        
        sigma = sigma.view(x.shape[0], 1).unsqueeze(-1).expand(-1, -1, x.shape[-1]).float()
        xx = torch.cat((sigma,x),dim=1)    
        return super(PRUNet,self).forward(xx)

def find_latest_checkpoint(output_dir):
    """Find the most recent checkpoint with proper numerical sorting"""
    import glob
    import re
    
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_epoch_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    def extract_epoch_batch(filename):
        # Extract epoch and batch from filename
        basename = os.path.basename(filename)
        match = re.search(r'checkpoint_epoch_(\d+)(?:_batch_(\d+))?\.pt', basename)
        if match:
            epoch = int(match.group(1))
            batch = int(match.group(2)) if match.group(2) else -1
            return (epoch, batch)
        return (0, -1)
    
    # Sort by epoch then by batch numerically
    latest = max(checkpoints, key=extract_epoch_batch)
    epoch, batch = extract_epoch_batch(latest)
    print(f"Found latest checkpoint: epoch {epoch}, batch {batch if batch != -1 else 'end'}")
    
    return latest

class Trainer:
    def __init__(self, model, dataloader, lr=1e-4, output_dir='./output'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.current_epoch = 0
        self.global_step = 0 
        self.train_losses = []
        self.learning_rates = []
        self.peak_memory = 0
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        self.output_dir = output_dir

    def training_step(self, batch):
        data_start = time.time()
        
        torch.cuda.empty_cache()
        sigma, noisy, clean = batch
        sigma = sigma.to(self.device).float()
        noisy = noisy.to(self.device).float()
        clean = clean.to(self.device).float()
        
        data_time = time.time() - data_start
        
        forward_start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        x_hat = self.model(noisy, sigma)
        
        forward_time = time.time() - forward_start
        
        backward_start = time.time()
        loss = self.criterion(x_hat, clean)
        if loss.item() > 100:
            print(f"Warning: High loss value detected: {loss.item()} at step {self.global_step}, sigma={sigma[0].item()}")
    
        loss.backward()
        del x_hat, noisy, clean, sigma

        # Aggressive cleanup every 50 steps
        if self.global_step % 50 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        backward_time = time.time() - backward_start
        
        if self.global_step % 10 == 0:
            print(f"Step {self.global_step} - Data: {data_time:.4f}s, Forward: {forward_time:.4f}s, Backward: {backward_time:.4f}s")
        
        self.global_step += 1
        
        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.peak_memory = max(self.peak_memory, current_memory)
        return loss.item()

    def save_checkpoint(self, epoch, batch_idx=None):
        """Save model checkpoint with zero-padded naming"""
        self.current_epoch = epoch  
        
        checkpoint = {
            'epoch': epoch + 1,  
            'batch_idx': batch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'learning_rates': self.learning_rates,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
        # Use checkpoints subdirectory with zero-padded naming
        checkpoint_dir = f"{self.output_dir}/checkpoints"
        if batch_idx is not None:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch:03d}_batch_{batch_idx:06d}.pt"
        else:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch:03d}.pt"
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Keep only the 3 most recent checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones"""
        import glob
        import re
        
        checkpoint_dir = f"{self.output_dir}/checkpoints"
        checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint_epoch_*.pt")
        
        def get_checkpoint_info(filename):
            basename = os.path.basename(filename)
            match = re.search(r'checkpoint_epoch_(\d+)(?:_batch_(\d+))?\.pt', basename)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2)) if match.group(2) else -1
                return (epoch, batch, filename)
            return (0, -1, filename)
        
        # Sort checkpoints by epoch and batch
        sorted_checkpoints = sorted([get_checkpoint_info(f) for f in checkpoint_files], 
                                   key=lambda x: (x[0], x[1]))
        
        # Keep only the 3 most recent
        if len(sorted_checkpoints) > 3:
            for epoch, batch, filepath in sorted_checkpoints[:-3]:
                os.remove(filepath)
                print(f"Removed old checkpoint: {os.path.basename(filepath)}")

    def train(self, start_epoch=0, start_batch=None, num_epochs=20):
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0
        
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0.0
            batches_processed = 0
            
            for i, batch in enumerate(self.dataloader):
                # Si on reprend au milieu d'une époque, ignorer les batches déjà traités
                if epoch == start_epoch and start_batch is not None and i <= start_batch:
                    continue
                    
                batch_loss = self.training_step(batch)
                epoch_loss += batch_loss
                batches_processed += 1
                
                if (i+1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(self.dataloader)}], "
                          f"Loss: {batch_loss:.6f}, Peak Memory: {self.peak_memory:.2f} MB")
                
                if (i+1) % 1000 == 0:
                    self.save_checkpoint(epoch, batch_idx=i)
            
            # Utiliser le compteur
            avg_train_loss = epoch_loss / max(batches_processed, 1)
            self.train_losses.append(avg_train_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1} completed - Train Loss: {avg_train_loss:.6f}")
            
            # Sauvegarder à la fin de chaque époque
            self.save_checkpoint(epoch)
            
            # Réinitialiser start_batch après la première époque
            if epoch == start_epoch:
                start_batch = None
                
            torch.cuda.empty_cache()
        
        print(f"Training completed in {(time.time()-start_time)/60:.2f} minutes")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint.get('batch_idx', None)
        
        # Restaurer les listes de métriques
        self.train_losses = checkpoint.get('train_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        # Restaurer l'état des générateurs aléatoires pour la reproductibilité
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'])
        if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        
        print(f"Checkpoint loaded from epoch {start_epoch}")
        if start_batch is not None:
            print(f"Resuming from batch {start_batch}")
        
        return start_epoch, start_batch

if __name__ == "__main__":
    import argparse
    from torch.utils.data import Subset
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--auto_resume', action='store_true', help='Automatically find latest checkpoint')
    parser.add_argument('--subset_size', type=int, default=1000000, help='Size of subset for testing')
    args = parser.parse_args()

    dataset = GRAND_H5_TracesDataset(
        h5_file_path='/sps/grand/selbouch/h5_files/grand_data.h5',
        num_realizations=2,
        snr_range_db=(-10, 20),  # SNR range in dB
        normalize=True,  # Enable normalization
        chunk_size=1000,    
        cache_size=500 
    )
    subset_indices = list(range(args.subset_size))
    train_subset = Subset(dataset, subset_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=128,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    model = PRUNet(in_nc=3, nb=2, nc=[64, 128, 256, 512, 1024])
    
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        lr=1e-3,
        output_dir='/sps/grand/selbouch/output'
    )

    # Gestion de la reprise
    start_epoch = 0
    start_batch = None
    
    if args.auto_resume:
        latest_checkpoint = find_latest_checkpoint('/sps/grand/selbouch/output/checkpoints')
        if latest_checkpoint:
            print(f"Auto-resuming from: {latest_checkpoint}")
            start_epoch, start_batch = trainer.load_checkpoint(latest_checkpoint)
        else:
            print("No checkpoint found, starting fresh training")
    elif args.resume:
        start_epoch, start_batch = trainer.load_checkpoint(args.resume)

    try:
        trainer.train(num_epochs=50, start_epoch=start_epoch, start_batch=start_batch)
    except KeyboardInterrupt:
        print("Checkpointing before exit...")
        trainer.save_checkpoint(trainer.current_epoch)
    
    torch.save(model.state_dict(), "/sps/grand/selbouch/output/final_model_1M.pth")
