import os
import sys
import argparse
from PIL import Image

import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms

from gradient_descent_network import *
from neumann_network import *


parser = argparse.ArgumentParser()
parser.add_argument('--testImage', required=True, default='test_sinogram.png')
parser.add_argument('--outdir', required=False, default='out', help='ckpt and output dir')
parser.add_argument('--blocks', type=int, default=6, dest='blocks', help='Number of blocks (iterations)')
parser.add_argument('--height', required=False, default=320, type=int)
parser.add_argument('--width', required=False, default=180, type=int)
parser.add_argument('--net', required=False, type=str, default='GD', help='GD: Unrolled Gradiant Descent; NN: Neumann Network')
parser.add_argument('--load', dest='load', type=int, required=True, default=-1, help='Load model from a .pth file by epoch #')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}.')

transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,)),
                              ])

image = Image.open(args.testImage)
assert image
image = transform(image)
image.unsqueeze_(0)
vutils.save_image(image, f'{args.outdir}/GT_sinogram.png', normalize=True)
print(image.shape)

try:
    if args.net == 'GD':
        m = GradientDescentNet(args=args, dataloader=None, device=device)
    elif args.net == 'NN':
        m = NeumannNet(args=args, dataloader=None, device=device)
    m.test(image)
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)