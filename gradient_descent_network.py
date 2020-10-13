import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from model import *
from utils import *

class GradientDescentNet():
    def __init__(self, args, dataloader, device):
        self.args = args
        self.dataloader = dataloader
        self.device = device
        
        self.resnet = nblock_resnet(n_residual_blocks=2).to(device)
        self.resnet = nn.DataParallel(self.resnet)
        if args.load < 0:
            self.resnet.apply(self.init_weights)
            self.start_epoch = 0
        else:
            self.load_checkpoints()
            self.start_epoch = args.load + 1
        
        self.eta = torch.tensor(0.1, dtype=torch.float32, requires_grad=True, device=device)
        self.opr = Operators(device=device)
        self.optimizer = optim.Adam(self.resnet.parameters(), lr=args.lr, betas=(0.999, 0.999))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        self.criterionL2 = torch.nn.MSELoss().to(device)
        
    
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) :
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.001)
    
    def load_checkpoints(self):
        path = os.path.join(self.args.outdir, 'ckpt/epoch'+str(self.args.load)+'.pth')
        self.resnet.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        print(f'Model loaded from {path}.')
    
    
    def run_block(self, beta):
#         print(torch.min(beta), torch.max(beta), torch.mean(beta))
#         sinogram = self.opr.radon.forward(beta)
#         print(torch.min(sinogram), torch.max(sinogram), torch.mean(sinogram))
#         x = self.opr.radon.backprojection(sinogram)
#         vutils.save_image(x.detach(), f'{self.args.outdir}/test.png', normalize=True)
        
#         x = self.opr.forward_gramian(beta)
#         print(torch.min(x), torch.max(x), torch.mean(x))
        linear_component = self.eta*self.opr.forward_gramian(beta) + self.network_input
        linear_component = beta - linear_component
        
        regulariser = self.resnet(beta)
        learned_component = -regulariser
        beta = linear_component + learned_component
        return beta
        
        
    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            for i, data in enumerate(self.dataloader):
                self.resnet.zero_grad()
                
                true_sinogram = data[0].to(self.device)
                true_beta = self.opr.FBP(true_sinogram)  # Ground Truth. Range=(0,1)
                
                self.network_input = self.opr.forward_adjoint(self.opr.undersample_model(true_sinogram))
                self.network_input *= self.eta
                beta = self.network_input

                for _ in range(self.args.blocks):
                    beta = self.run_block(beta)
                
                self.err = self.criterionL2(beta, true_beta)
                self.err.backward()
                self.optimizer.step()
#                 self.scheduler.step()
                
                if i % 100 == 0:
                    self.log(epoch, i)
                    
            self.log(epoch, i)
            torch.save(self.resnet.state_dict(), f'{self.args.outdir}/ckpt/epoch{epoch}.pth')
            vutils.save_image(beta.detach(), f'{self.args.outdir}/result_samples_epoch{epoch}.png', normalize=True)

            
    def log(self, epoch, i):
        print(f'[{epoch}/{self.args.epochs}][{i}/{len(self.dataloader)}] ' \
              f'lr:{self.scheduler.get_last_lr()[0]:.4f} ' \
              f'eta:{self.eta.item():.4f} ' \
              f'Loss:{self.err.item():.4f} ' \
             )