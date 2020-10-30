import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from model import *
from utils import *


class NeumannNet():
    def __init__(self, args, dataloader, device):
        self.args = args
        self.dataloader = dataloader
        self.device = device
        self.opr = Operators(args.height, args.width, 8, device=device)
        
        self.resnet = nblock_resnet(n_residual_blocks=2).to(device)
        self.resnet = nn.DataParallel(self.resnet)
        if args.load < 0:
            self.resnet.apply(self.init_weights)
            self.start_epoch = 0
            self.eta = self.opr.estimate_eta()  #.requires_grad_() # uncomment to train eta
            print(f'initial eta estimate: {self.eta:.6f}')
        else:
            self.load_checkpoints()
            self.start_epoch = args.load + 1
        
        
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) :
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.001)
            
            
    def load_checkpoints(self):
        resnet_path = os.path.join(self.args.outdir, 'ckpt/resnet_epoch'+str(self.args.load)+'.pth')
        self.resnet.load_state_dict(torch.load(resnet_path, map_location=self.device))
        para_path = os.path.join(self.args.outdir, 'ckpt/parameters_epoch'+str(self.args.load)+'.pth')
        paras = torch.load(para_path, map_location=self.device)
        self.eta = paras['eta']  #.requires_grad_() # uncomment to train eta
        self.args.lr = paras['lr']
        print(f'Model loaded from {resnet_path} and {para_path}.')
        print(f'eta starts from {self.eta.item()}')
        print(f'lr starts from {self.args.lr}')
        
        
    def run_block(self, beta):
        linear_component = beta - self.eta*self.opr.forward_gramian(beta)
        regulariser = self.resnet(beta)
        learned_component = -regulariser
        beta = linear_component + learned_component
        return linear_component
    
    
    def train(self):
        '''
            Train Phase
        '''
        self.optimizer = optim.Adam([{"params":self.resnet.parameters()}
#                                      ,{"params":self.eta}  # uncomment to train eta
                                    ], lr=self.args.lr, betas=(0.999, 0.999))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.97)
        self.criterionL2 = torch.nn.MSELoss().to(self.device)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            for i, data in enumerate(self.dataloader):
                self.resnet.zero_grad()
                
                true_sinogram = data[0].to(self.device)
                true_beta = self.opr.FBP(true_sinogram)  # Ground Truth Reconstruction
                
                self.network_input = self.opr.forward_adjoint(self.opr.undersample_model(true_sinogram))
                self.network_input *= self.eta
                beta = self.network_input
                self.neumann_sum = beta
                
                for _ in range(self.args.blocks):  # run iterations
                    beta = self.run_block(beta)
                    self.neumann_sum += beta
                    
                self.err = self.criterionL2(self.neumann_sum, true_beta)
                self.err.backward()
                self.optimizer.step()  # update parameters
                
                if i % 100 == 0:
                    self.log(epoch, i)
#             break
            self.log(epoch, i)
            
            torch.save(self.resnet.state_dict(), f'{self.args.outdir}/ckpt/resnet_epoch{epoch}.pth')
            torch.save({'eta':self.eta, 'lr':self.scheduler.get_last_lr()[0]}, 
                       f'{self.args.outdir}/ckpt/parameters_epoch{epoch}.pth')
            vutils.save_image(self.neumann_sum.detach(), f'{self.args.outdir}/train_samples_epoch{epoch}.png', normalize=True)
            self.scheduler.step()  # update learning rate, disable this if no exp decay
            
    
    def test(self, true_sinogram, eta):
        '''
            Test Phase (for single test image).
        '''
        self.network_input = self.opr.forward_adjoint(self.opr.undersample_model(true_sinogram))
        self.network_input *= self.eta
        beta = self.network_input
        self.neumann_sum = beta

        for _ in range(self.args.blocks):  # run iterations
            beta = self.run_block(beta)
            self.neumann_sum += beta

        vutils.save_image(self.neumann_sum, f'{self.args.outdir}/test_result.png', normalize=True)
        
            
    def log(self, epoch, i):
        print(f'[{epoch}/{self.args.epochs}][{i}/{len(self.dataloader)}] ' \
              f'lr:{self.scheduler.get_last_lr()[0]} ' \
              f'eta:{self.eta.item()} ' \
              f'Loss:{self.err.item()} ' \
             )