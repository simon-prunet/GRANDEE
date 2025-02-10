import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import grand.dataio.root_files as froot
from torch import Tensor, IntTensor
from torch.nn.functional import pad
import glob

#import matplotlib.pyplot as plt

class GRAND_DC2_TracesDataset(Dataset):

    '''
    A pytorch data loader for GRAND DC2 simulated EAS traces 
    '''
    def __init__(self,rootpath='/sps/grand/DC2Training/ZHAireS',dataset='NJ', what='efield', level='L0', 
                 transform='pad', n_antennas=300):
        self.datadir=rootpath+'-'+dataset
        self.transform = transform
        self.what = what
        self.level = level
        self.na = n_antennas

    def __len__(self):
        '''
        12 files contain each 1000 events with associated traces
        '''
        return (12000)

    def __getitem__(self,idx, return_extra=False):
        '''
        Get the traces out, as a numpy array of shape (#antennas, #samples, #polars).
        Eventually gets transformed to a pytorch tensor if transform=ToTensor
        '''
        filename = self._get_file_name(idx)
        # Get EAS parameters dictionary
        parameters = froot.get_simu_parameters(filename, idx%1000)
        # Get EAS traces
        ef3d = froot.get_handling3dtraces(filename,idx%1000)
        traces = ef3d.traces
        triggered_antenna_indices = ef3d.idx2idt
        if (self.transform=='pad'):
            traces = pad(Tensor(traces),(0,0,0,0,0,self.na - traces.shape[0]),value=0.)
            triggered_antenna_indices = pad(IntTensor(triggered_antenna_indices),(0,self.na - triggered_antenna_indices.shape[0]),value=-1)
        if (return_extra):
            return (traces, triggered_antenna_indices, parameters)
        else:
            return (traces, triggered_antenna_indices)

    def _get_file_name(self, index):
        '''
        12 files contain each 1000 events with associated traces
        '''
        num_file = index//1000
        fileroot = 'sim_Xiaodushan_20221025_220000_RUN0_CD_ZHAireS_'
        filename = fileroot+str(num_file).rjust(4,'0')
        efield_file = glob.glob(self.datadir+'/'+filename+'/'+self.what+'*'+self.level+'*')
        if len(efield_file)==0 :
            print("File not found")
            return None
        
        return (efield_file[0])

    