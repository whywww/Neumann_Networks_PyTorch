import torch
import numpy as np
from torch_radon import Radon


class Operators():
    def __init__(self, device):
        self.image_size = 320
        self.n_angles = 180
        angles = np.linspace(0, np.pi, self.n_angles, endpoint=False)
        self.radon = Radon(self.image_size, angles, clip_to_circle=True)
        sample_rate = 8
        self.mask = torch.zeros((1,1,1,180)).to(device)
        self.mask[:,:,:,::sample_rate].fill_(1)
        
        
    # $X^\T ()$  inverse radon with $|X^T|_{op}=1$
    def forward_adjoint(self, input):
        # check dimension
        if input.size()[2] != self.image_size or input.size()[3] != self.n_angles:
            raise Exception(f'forward_adjoint input dimension wrong! received {input.size()}.')
            
        return self.radon.backprojection(input.permute(0,1,3,2))//self.n_angles
    

    # $X^\T X ()$ with $|X^T|_{op}=1$
    def forward_gramian(self, input):
        # check dimension
        if input.size()[2] != self.image_size:
            raise Exception(f'forward_gramian input dimension wrong! received {input.size()}.')
            
        sinogram = self.radon.forward(input)
        bp = self.radon.backprojection(sinogram)/self.n_angles
        return normalize(bp)
    

    # corruption model: undersample sinogram by 8
    def undersample_model(self, input):
        return input*self.mask
    
    
    # Filtered Backprojection. Input siogram range = (0,1)
    def FBP(self, input):
        # check dimension
        if input.size()[2] != self.image_size or input.size()[3] != self.n_angles:
            raise Exception(f'FBP input dimension wrong! received {input.size()}.')
        filtered_sinogram = self.radon.filter_sinogram(input.permute(0,1,3,2))
        fbp = self.radon.backprojection(filtered_sinogram)/self.n_angles
        return normalize(fbp)
    
    
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