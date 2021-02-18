import torch
import numpy as np
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import argparse

from utils import *
from gradient_descent_network import *
from neumann_network import *


parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, default='data', help='directory to sinogram dataset')
parser.add_argument('--ckptdir', required=False, default='out', help='ckpt dir')
parser.add_argument('--saveto', required=False, default='out', help='output dir')
parser.add_argument('--blocks', type=int, default=6, dest='blocks', help='Number of blocks (iterations)')
parser.add_argument('--bs', type=int, default=10, help='Batch size')
parser.add_argument('--net', required=False, type=str, default='GD', help='GD: Unrolled Gradiant Descent; NN: Neumann Network')
parser.add_argument('--beam', required=False, type=str, default='parallel', help='parallel: parallel beam; fan: fan beam.')
parser.add_argument('--size', required=False, type=int, default=320, help='the size of the input image to network.')
parser.add_argument('--angles', required=False, type=int, default=180, help='full-view projection angles.')
parser.add_argument('--det_size', required=False, type=int, help='detector pixel number, default: image size.')
parser.add_argument('--rate', type=int, default=8, help='undersample rate')
parser.add_argument('--load', dest='load', type=int, required=True, default=-1, help='Load model from a .pth file by epoch #')
parser.add_argument('--class_name', required=False, type=str, default='C')
args = parser.parse_args()


def batch_test(net):
    for i, data in enumerate(m.dataloader):
        y = data[0].to(device)
        results = m.test(y)
        print(f'Saving images for batch {i}')
        for j in range(y.size()[0]):
            vutils.save_image(results[j,0], f'{args.saveto}/{class_name}/{fnames[i*args.bs+j]}', normalize=True)  # to 0~255

            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.datadir is None:
    raise ValueError("`datadir` parameter is required for dataset")

dataset = dset.ImageFolder(root=args.datadir,
                            transform=transforms.Compose([
                                transforms.Resize((args.size,args.size)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,)),  # has already been 0~1 before this, to -1~1
                            ]))

assert dataset

class_name = args.class_name
class_idx = dataset.class_to_idx[class_name]
print(class_name, class_idx)
targets = torch.tensor(dataset.targets)
target_idx = np.nonzero(targets == class_idx)
print(len(target_idx))
subset = torch.utils.data.Subset(dataset, target_idx)
sampler = torch.utils.data.sampler.SequentialSampler(subset)
dataloader = torch.utils.data.DataLoader(subset, sampler=sampler, batch_size=args.bs)

# make sure file sequence is the same
fnames = sorted(sorted(os.walk(os.path.join(args.datadir, class_name), followlinks=True))[0][2])

try:
    if not os.path.exists(os.path.join(args.saveto, class_name)):
        os.makedirs(os.path.join(args.saveto, class_name))
        print(f'Created {args.saveto} directory')
    else:
        print('saving to', os.path.join(args.saveto, class_name))
except OSError:
    pass

args.det_size = args.size if args.det_size == None else args.det_size

try:
    if args.net == 'GD':
        m = GradientDescentNet(args=args, dataloader=dataloader, device=device)
    elif args.net == 'NN':
        m = NeumannNet(args=args, dataloader=dataloader, device=device)
    batch_test(m)
    
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)