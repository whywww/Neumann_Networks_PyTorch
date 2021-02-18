# Neumann Networks Implementation in PyTorch

Original Tensorflow version: https://github.com/dgilton/neumann_networks_code

Paper: https://arxiv.org/abs/1901.03707

- Unrolled Gradient Descent Network and Neumann Network implemented for sparse-view CT reconstruction.
- Both parallel and fan beam supported.

Requirements: 
- Reconstruction domain dataset
- [TorchRadon](https://github.com/matteo-ronchetti/torch-radon)

#### Example train command:

- parallel beam

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--datadir path/to/train/dataset --ckptdir out_NN_rate8 \
--bs 10 --net NN --eta 0.1 --rate 8 \
--beam parallel --size 320 --angles 180 \
--load -1
```

- fan beam

Please also specify `det_size, angles, src_dist` and `det_dist` for fan beam.

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
--datadir path/to/train/dataset --ckptdir out_NN_fan_rate8 \
--bs 10 --net NN --eta 0.1 --rate 8 \
--beam fan --det_size 480 --angles 208 \
--load -1
```

#### Example test command:

- single predict

```
CUDA_VISIBLE_DEVICES=0 python3 predict.py \
--ckptdir out_NN_rate8 --net NN --rate 8 \
--beam parallel --size 320 --angles 180 \
--testImage test_image.png \
--saveas reconstruction_result.png \
--load 99
```

- batch predict

Test images should be in `path/to/test/dataset/{class_name}`, and results are saved to `{saveto}/{class_name}`:

```
CUDA_VISIBLE_DEVICES=0 python3 predict_all.py \
--ckptdir out_NN_rate8 --net NN --rate 8 \
--datadir path/to/test/dataset \
--saveto path/to/save/results \
--class_name N \
--load 99
```