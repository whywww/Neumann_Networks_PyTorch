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
parser.add_argument('--testImage', required=True, default='test_image.png')
parser.add_argument('--ckptdir', required=False, default='out', help='ckpt and output dir')
parser.add_argument('--saveas', required=False, default='test_result.png', help='save path, end with .png')
parser.add_argument('--blocks', type=int, default=6, dest='blocks', help='Number of blocks (iterations)')
parser.add_argument('--beam', required=False, type=str, default='parallel', help='parallel: parallel beam; fan: fan beam.')
parser.add_argument('--size', required=False, type=int, default=320, help='the size of the input image to network.')
parser.add_argument('--angles', required=False, type=int, default=180, help='full-view projection angles.')
parser.add_argument('--det_size', required=False, type=int, help='detector pixel number, default: image size.')
parser.add_argument('--rate', type=int, default=8, help='undersample rate')
parser.add_argument('--net', required=False, type=str, default='GD', help='GD: Unrolled Gradiant Descent; NN: Neumann Network')
parser.add_argument('--load', dest='load', type=int, required=True, default=-1, help='Load model from a .pth file by epoch #')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}.')

transform = transforms.Compose([
                                transforms.Resize((args.size,args.size)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,)),
                              ])

image = Image.open(args.testImage)
assert image
image = transform(image)
image.unsqueeze_(0)

args.det_size = args.size if args.det_size == None else args.det_size

try:
    if not os.path.exists(os.path.dirname(args.saveas)):
        os.makedirs(os.path.dirname(args.saveas))
except OSError:
    pass


try:
    if args.net == 'GD':
        m = GradientDescentNet(args=args, dataloader=None, device=device)
    elif args.net == 'NN':
        m = NeumannNet(args=args, dataloader=None, device=device)
    result = m.test(image.to(device))
    vutils.save_image(result, f'{args.saveas}', normalize=True)
    
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)