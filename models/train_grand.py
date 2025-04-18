import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import grand.dataio.root_files as froot
import glob
import numpy as np
from functools import lru_cache
import time
import logging
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import basicblock as B

class GRAND_DC2_TracesDataset(Dataset):
    def __init__(self, rootpath='/sps/grand/DC2Training/ZHAireS', dataset='NJ', 
                 what='efield', level='L0', num_realizations=5, 
                 trace_length=8192, noise_range=(0.1, 10), normalize=True):
        self.datadir = f"{rootpath}-{dataset}"
        self.what = what
        self.level = level
        self.num_realizations = num_realizations
        self.trace_length = trace_length
        self.noise_range = noise_range
        self.normalize = normalize
        self.trace_indices = []
        self._precompute_indices() 
        
    def _precompute_indices(self):
        """Precompute all accessible trace indices with realizations"""
        base_indices = []
        for event_idx in range(6000):  
            try:
                ef3d = self._load_event_data(event_idx)
                n_antennas = ef3d.traces.shape[0]
                base_indices.extend((event_idx, ant_idx) 
                                 for ant_idx in range(n_antennas))
            except FileNotFoundError:
                continue
            
        self.trace_indices = [
            (e, a, r)
            for e, a in base_indices
            for r in range(self.num_realizations)
        ]
        
        print(f"Dataset: {len(base_indices)} unique traces → "
              f"{len(self.trace_indices)} total items")

    @lru_cache(maxsize=100) 
    def _load_event_data(self, event_idx):
        """Cached event data loading"""
        filename = self._get_file_name(event_idx)
        return froot.get_handling3dtraces(filename, event_idx % 1000)

    def __len__(self):
        return len(self.trace_indices)
    
    def __getitem__(self, idx):
        event_idx, ant_idx, realization_idx = self.trace_indices[idx]
        
        clean_trace = self._load_event_data(event_idx).traces[ant_idx]
        
        if self.normalize:
            max_val = np.max(np.abs(clean_trace)) + 1e-8
            clean_trace = clean_trace / max_val
            
        clean_tensor = torch.as_tensor(clean_trace, dtype=torch.float32)
        
        sigma = torch.empty(1).uniform_(*self.noise_range).item()
        noise = torch.randn_like(clean_tensor) * sigma
        
        return (
            torch.tensor(sigma, dtype=torch.float32),
            clean_tensor + noise,
            clean_tensor
        )
            
    def _get_file_name(self, index):
        num_file = index // 1000
        fileroot = 'sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_'
        filename = f"{fileroot}{num_file:04d}"
        efield_files = glob.glob(f"{self.datadir}/{filename}/{self.what}*{self.level}*")
        
        if not efield_files:
            raise FileNotFoundError(f"No {self.level} files found for index {index}")
        return efield_files[0]

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
        
        # upsample
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

    def __init__(self, in_nc=3, n_downs=3, nc=[64, 128, 256, 512], nb=2):

        super(PRUNet,self).__init__(in_nc=in_nc+1, out_nc=in_nc, n_downs=n_downs, nc=nc, nb=nb)

    
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


class Trainer:
    def __init__(self, model, dataloader, val_dataloader=None, lr=1e-4, output_dir='./output'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.current_epoch = 0
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.peak_memory = 0
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        self.output_dir = output_dir
        self.best_val_loss = float('inf')

    def training_step(self, batch):
        torch.cuda.empty_cache()
        sigma, noisy, clean = batch
        sigma = sigma.to(self.device).float()
        noisy = noisy.to(self.device).float()
        clean = clean.to(self.device).float()
        
        self.optimizer.zero_grad(set_to_none=True)
        x_hat = self.model(noisy, sigma)
        loss = self.criterion(x_hat, clean)
        loss.backward()
        self.optimizer.step()

        current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.peak_memory = max(self.peak_memory, current_memory)
        return loss.item()

    def train(self, start_epoch=0, num_epochs=20, patience=10):
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        no_improve_epochs = 0
        
        torch.cuda.reset_peak_memory_stats()
        self.peak_memory = 0
        
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for i, batch in enumerate(self.dataloader):
                batch_loss = self.training_step(batch)
                epoch_loss += batch_loss
                
                if (i+1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(self.dataloader)}], "
                          f"Loss: {batch_loss:.6f}, Peak Memory: {self.peak_memory:.2f} MB")
            
            avg_train_loss = epoch_loss / len(self.dataloader)
            self.train_losses.append(avg_train_loss)
            

            torch.cuda.empty_cache()
            
            val_loss = self._validate()
            self.val_losses.append(val_loss if val_loss is not None else 0)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1} completed - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f if val_loss is not None else 'N/A'}, "
                  f"Peak Memory: {self.peak_memory:.2f} MB")
            
            if val_loss is not None:
                self.scheduler.step(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                
                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
                self.plot_progress()
                torch.cuda.empty_cache()
        
        print(f"Training completed in {(time.time()-start_time)/60:.2f} minutes")
        print(f"Peak memory usage: {self.peak_memory:.2f} MB")
        self.plot_progress()

    def _validate(self):
        """Validate the model on the validation dataset"""
        if not self.val_dataloader:
            return None
            
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                sigma, noisy, clean = batch
                sigma = sigma.to(self.device).float()
                noisy = noisy.to(self.device).float()
                clean = clean.to(self.device).float()
                
                x_hat = self.model(noisy, sigma)
                loss = self.criterion(x_hat, clean)
                val_loss += loss.item()
        
        return val_loss / len(self.val_dataloader)

    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']  
        print(f"Checkpoint chargé (epoch {start_epoch})")
        return start_epoch  
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        self.current_epoch = epoch  
        
        checkpoint = {
            'epoch': epoch + 1,  
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        
    
        checkpoint_path = f"{self.output_dir}/checkpoints/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = f"{self.output_dir}/checkpoints/best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
            
        checkpoint_files = sorted(glob.glob(f"{self.output_dir}/checkpoints/checkpoint_epoch_*.pt"))
        if len(checkpoint_files) > 5:  # Keep only the 5 most recent checkpoints
            for old_file in checkpoint_files[:-5]:
                os.remove(old_file)



    def plot_progress(self):
        """Plot and save training progress"""
        plt.figure(figsize=(15, 10))
        

        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.yscale('log')
        
    
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_progress.png")
        plt.close()
        
        import pandas as pd
        df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if self.val_losses else [0] * len(self.train_losses),
            'learning_rate': self.learning_rates
        })
        df.to_csv(f"{self.output_dir}/training_metrics.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    args = parser.parse_args()
  

    dataset = GRAND_DC2_TracesDataset(
        rootpath='/sps/grand/DC2Training/ZHAireS',
        num_realizations=2,
        noise_range=(0.1, 3),
    )
    

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  
        shuffle=True,
        num_workers=1,  
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    """val_loader = DataLoader(
        val_dataset,
        batch_size=64,  
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=True
    )"""


    model = PRUNet(in_nc=3)
    

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        val_dataloader=None,
        lr=1e-3,
        output_dir='./output'
    )

    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)  
    else:
        start_epoch = 0  

    try:
        trainer.train(num_epochs=100,start_epoch=start_epoch) 
    except KeyboardInterrupt:
        print("Checkpointing before exit...")
        trainer.save_checkpoint(trainer.current_epoch)
    
    torch.save(model.state_dict(), "./output/final_model.pt")
