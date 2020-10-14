import os
from PIL import Image

import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms

from gradient_descent_network import *


parser = argparse.ArgumentParser()
parser.add_argument('--testImage', required=False, default='test_sinogram.png')
parser.add_argument('--outdir', required=False, default='out', help='ckpt and output dir')
parser.add_argument('--height', required=False, default=320, type=int)
parser.add_argument('--width', required=False, default=180, type=int)
parser.add_argument('--load', dest='load', type=int, required=True, default=-1, help='Load model from a .pth file by epoch #')
args = parser.parse_args()


transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.Grayscale(), # 1 channel
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

image = Image.open(args.testImage)
image = transform(image)
image.unsqueeze_(0)
vutils.save_image(image, f'{self.outdir}/GT_sinogram.png', normalize=True)


try:
    m = GradientDescentNet(args=args, dataloader=dataloader, device=device)
    m.test()
except KeyboardInterrupt:
    print('Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)