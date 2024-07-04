# CS-VGL

Cross-domain Switch-aware Re-parameterization for Visual Geo-Localization

```shell
CUDA_VISIBLE_DEVICES=0 python train.py --train_batch_size 96 --aggregation netvlad --backbone dinov2_vitl14 --use_amp16 --dataset_name pitts30k --epochs_num 10 --resize_test_imgs --use_cls --clusters 32 --trainable_layers 19,20,21,22,23


ython train.py --train_batch_size 64 --aggregation netvlad --backbone dinov2_vitl14 --use_amp16 --dataset_name msls --epochs_num 20 --resize_test_imgs --use_cls --clusters 32 --trainable_layers 19,20,21,22,23 --use_extra_datasets --resume logs/dinov2_vitl14_netvlad/gsv_cities/tokenvlad_k32_l256_G/best_model.pth --lr 0.000003 --resize 448 448 --num_workers 32 --infer_batch_size 256
``