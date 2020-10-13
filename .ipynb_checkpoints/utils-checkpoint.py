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
        
        
    # $X^\T ()$  inverse radon
    def forward_adjoint(self, input):
        # check dimension
        if input.size()[2] != self.image_size or input.size()[3] != self.n_angles:
            raise Exception(f'forward_adjoint input dimension wrong! received {input.size()}.')
            
        filtered_sinogram = self.radon.filter_sinogram(input)
        return self.radon.backprojection(filtered_sinogram)
    

    # $X^\T X ()$
    def forward_gramian(self, input):
        # check dimension
        if input.size()[2] != self.image_size:
            raise Exception('forward_gramian input dimension wrong!')
            
        sinogram = self.radon.forward(input)
        filtered_sinogram = self.radon.filter_sinogram(sinogram)
        return self.radon.backprojection(filtered_sinogram)
    

    # corruption model: undersample sinogram by 8
    def undersample_model(self, input):
        return input*self.mask
    