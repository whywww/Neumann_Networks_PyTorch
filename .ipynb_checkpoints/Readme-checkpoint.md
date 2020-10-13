# Neumann Networks Implementation in PyTorch

Unrolled Gradient Descent for CT reconstruction implemented.

Example run command:

```
CUDA_VISIBLE_DEVICES=1,2 python3 main.py \
--bs 32 --outdir out_GD \
--datadir data \
--load -1
```