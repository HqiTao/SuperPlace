# CS-VGL

Cross-domain Switch-aware Re-parameterization for Visual Geo-Localization

## G2M (ReLU)

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 3

CUDA_VISIBLE_DEVICES=1 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 32

CUDA_VISIBLE_DEVICES=2 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 64

CUDA_VISIBLE_DEVICES=3 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 128
```

## G2M (GELU)

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 3

CUDA_VISIBLE_DEVICES=1 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 32

CUDA_VISIBLE_DEVICES=2 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 64

CUDA_VISIBLE_DEVICES=3 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 128
```

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --use_cls --clusters 32 --trainable_layers 19,20,21,22,23 --use_extra_datasets --resume logs/dinov2_vitl14_netvlad/gsv_cities/tokenvlad_k32_l256_G/best_model.pth --lr 0.000003 --resize 448 448 --num_workers 32 --infer_batch_size 256
```