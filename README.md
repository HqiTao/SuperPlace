# SuperPlace



## Experiments

```shell
# G2M
python train.py --train_batch_size 64 --aggregation g2m --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_ca --num_hiddens 64
# NetVLAD
python train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --clusters 64
# NVL-Linear
python train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitb14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_cls --clusters 64 --linear_dim 128 --use_linear
# DINO-Large-NetVLAD
python train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitl14 --use_amp16 --dataset_name msls --epochs_num 5 --resize_test_imgs --infer_batch_size 64 --use_extra_datasets --trainable_layer 19,20,21,22,23
# Eval
python eval.py --aggregation g2m --backbone dinov2_vitb14 --dataset_name tokyo247 --infer_batch_size 256 --use_ca --num_hiddens 64 --resume logs/path
```
