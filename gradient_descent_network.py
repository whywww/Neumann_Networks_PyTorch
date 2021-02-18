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
        print(f'Undersample rate is: {args.rate}')
        
        if args.beam == 'parallel':
            self.opr = Operators(args.size, args.angles, args.rate, device=device)
        elif args.beam == 'fan':
            self.opr = Operators(args.size, args.angles, args.rate, device=device, beam=args.beam, det_size=args.det_size)
        else:
            raise Exception('projection beam type wrong!')
        
        self.resnet = nn.DataParallel(nblock_resnet(n_residual_blocks=2).to(device))
        self.init_network()
        
        
    def init_network(self):
        if self.args.load < 0:
            self.resnet.apply(self.init_weights)
            self.start_epoch = 0
            self.eta = self.args.eta if self.args.eta != None else self.opr.estimate_eta()  #.requires_grad_() # uncomment to train eta
            print(f'initial eta estimate: {self.eta:.6f}')
        else:
            self.load_checkpoints()
            self.start_epoch = self.args.load + 1
            
            
    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) :
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.001)

            
    def load_checkpoints(self):
        resnet_path = os.path.join(self.args.ckptdir, 'ckpt/resnet_epoch'+str(self.args.load)+'.pth')
        self.resnet.load_state_dict(torch.load(resnet_path, map_location=self.device))
        para_path = os.path.join(self.args.ckptdir, 'ckpt/parameters_epoch'+str(self.args.load)+'.pth')
        paras = torch.load(para_path, map_location=self.device)
        self.eta = paras['eta']  #.requires_grad_() # uncomment to train eta
        self.args.lr = paras['lr']
        print(f'Model loaded from {resnet_path} and {para_path}.')
        print(f'eta starts from {self.eta}')
        print(f'lr starts from {self.args.lr}')
        
    
    def run_block(self, beta):
        linear_component = beta - self.eta*self.opr.forward_gramian(beta) + self.network_input
        regulariser = self.resnet(beta)
        learned_component = -regulariser*self.eta
        beta = linear_component + learned_component
        return beta
        
        
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
                
                true_beta = data[0].to(self.device)  # Ground Truth Reconstruction 0~1
                true_sinogram = self.opr.forward_radon(true_beta)  # 0~1
                
                self.network_input = self.opr.forward_adjoint(self.opr.undersample_model(true_sinogram))
                self.network_input *= self.eta
                beta = self.network_input

                for _ in range(self.args.blocks):  # run iterations
                    beta = self.run_block(beta)
 
                self.err = self.criterionL2(beta, true_beta)
                self.err.backward()
                self.optimizer.step()  # update parameters
                
                if i % 100 == 0:
                    self.log(epoch, i)

            self.log(epoch, i)
            
            torch.save(self.resnet.state_dict(), f'{self.args.ckptdir}/ckpt/resnet_epoch{epoch}.pth')
            torch.save({'eta':self.eta, 'lr':self.scheduler.get_last_lr()[0]}, 
                       f'{self.args.ckptdir}/ckpt/parameters_epoch{epoch}.pth')
            vutils.save_image(beta.detach(), f'{self.args.ckptdir}/train_samples_epoch{epoch}.png', normalize=True)
            self.scheduler.step()  # update learning rate, disable this if no exp decay
            
            
    def test(self, true_beta):
        '''
            Test Phase (for single test image).
        '''
        true_sinogram = self.opr.forward_radon(true_beta)  # 0~1
        self.network_input = self.opr.forward_adjoint(self.opr.undersample_model(true_sinogram.to(self.device)))
        self.network_input *= self.eta
        beta = self.network_input

        for _ in range(self.args.blocks):  # run iterations
            beta = self.run_block(beta)
        
        return beta.detach()
        
            
    def log(self, epoch, i):
        print(f'[{epoch}/{self.args.epochs}][{i}/{len(self.dataloader)}] ' \
              f'lr:{self.scheduler.get_last_lr()[0]} ' \
#               f'eta:{self.eta.item()} ' \
              f'Loss:{self.err.item()} ' \
             )