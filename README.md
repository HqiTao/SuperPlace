# SuperPlace

Super Alignment, Embodied Re-ranking and Two Improved Aggregation for Visual Place Recognition



## G2M (ReLU) (GELU)

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 3

CUDA_VISIBLE_DEVICES=1 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 32

CUDA_VISIBLE_DEVICES=2 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 64

CUDA_VISIBLE_DEVICES=3 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 128
```

## GeM

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64
```

## G2M CLS

```shell
CUDA_VISIBLE_DEVICES=1 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 64 --use_cls --features_dim 1536
```

## GeM SE

```shell
CUDA_VISIBLE_DEVICES=2 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 64 --ca_method avg
```

## GeM CBA

```shell
CUDA_VISIBLE_DEVICES=3 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 64 --ca_method cba
```


## NeVLAD
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --clusters 32
CUDA_VISIBLE_DEVICES=1 python train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --clusters 64
```

## SALAD
```shell
CUDA_VISIBLE_DEVICES=3 python train.py --train_batch_size 64 --aggregation salad --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --features_dim 8448
```

## TokenVLAD
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_cls --clusters 32 --linear_dim 256
CUDA_VISIBLE_DEVICES=1 python train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_cls --clusters 64 --linear_dim 128
```