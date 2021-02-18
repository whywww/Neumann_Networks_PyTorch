import torch
import numpy as np
from torch_radon import Radon, RadonFanbeam
from torch_radon.solvers import Landweber


class Operators():
    def __init__(self, img_size, n_angles, sample_ratio, device, circle=False, beam='parallel', det_size=None, det_dist=(1800, 0)):
        self.device = device
        self.img_size = img_size
        self.sample_ratio = sample_ratio
        self.n_angles = n_angles
        self.det_size = img_size if det_size == None else det_size
        
        self.init_radon(beam, circle, det_dist)
        self.landweber = Landweber(self.radon)
        
        self.mask = torch.zeros((1,1,1,n_angles)).to(device)
        self.mask[:,:,:,::sample_ratio].fill_(1)
        
        
    def init_radon(self, beam, circle, det_dist):
        if beam == 'parallel':
            angles = np.linspace(0, np.pi, self.n_angles, endpoint=False)
            self.radon = Radon(self.img_size, angles, clip_to_circle=circle)
            self.radon_sparse = Radon(self.img_size, angles[::self.sample_ratio], clip_to_circle=circle)
        elif beam == 'fan':
            angles = np.linspace(0, self.n_angles/180*np.pi, self.n_angles, False)
            self.radon = RadonFanbeam(self.img_size, angles, source_distance=det_dist[0], det_distance=det_dist[1], 
                                      clip_to_circle=circle, det_count=self.det_size)            
            self.radon_sparse = RadonFanbeam(self.img_size, angles[::self.sample_ratio], source_distance=det_dist[0], 
                                             det_distance=det_dist[1], clip_to_circle=circle, det_count=self.det_size)
        else:
            raise Exception('projection beam type undefined!')
        self.n_angles_sparse = len(angles[::self.sample_ratio])
    
    
    # Radon Transform. 
    def forward_radon(self, input):
        # check dimension
        if list(input.shape[2:]) != [self.img_size, self.img_size]:
            raise Exception(f'radon input dimension wrong! received {input.size()}.')
        return self.radon.forward(input)/self.det_size
    
    
    # $X^\T ()$ inverse radon
    def forward_adjoint(self, input):
        # check dimension
        if list(input.shape[2:]) == [self.n_angles, self.det_size]:
            return self.radon.backprojection(input)/self.n_angles
        elif list(input.shape[2:]) == [self.n_angles_sparse, self.det_size]:
            return self.radon_sparse.backprojection(input)/self.n_angles_sparse  # scale the angles
        else:
            raise Exception(f'forward_adjoint input dimension wrong! received {input.size()}.') 
        
        
    # $X^\T X ()$
    def forward_gramian(self, input):
        sinogram = self.forward_radon(input)
        return self.forward_adjoint(sinogram)
    

    # Corruption model: undersample sinogram by sample_ratio
    def undersample_model(self, input):
        return input[:,:,::self.sample_ratio,:]


    # estimate step size eta
    def estimate_eta(self):
        eta = self.landweber.estimate_alpha(self.img_size, self.device)
        return torch.tensor(eta, dtype=torch.float32, device=self.device)
    

def normalize(x, rfrom=None, rto=(0,1)):
    if rfrom is None:
        mean = torch.tensor([torch.min(x),]).cuda()
        std = torch.tensor([(torch.max(x)-torch.min(x)),]).cuda()
        x = x.sub(mean[None, :, None, None]).div(std[None, :, None, None]).mul(rto[1]-rto[0]).add(rto[0])
    else:
        mean = torch.tensor([rfrom[0],]).cuda()
        std = torch.tensor([rfrom[1]-rfrom[0],]).cuda()
        x = x.sub(mean[None, :, None, None]).div(std[None, :, None, None]).mul(rto[1]-rto[0]).add(rto[0])
    return x