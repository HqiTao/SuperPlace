# CS-VGL

Cross-domain Switch-aware Re-parameterization for Visual Geo-Localization

```shell
CUDA_VISIBLE_DEVICES=1,2 python train.py --train_batch_size 128  --epochs_num 10  --features_dim 768  --aggregation cosgem --backbone dinov2_vitb14 --use_amp16 --resize_test_imgs
```

1.Generalized Channel Attention

|      Setups     | Pitts | MSLS | 
|       ----      | ----  | ---- |
|  GeM (baseline) | 92.1  | 88.2 |
|  GeM + SE       | 92.0  | 88.0 |
|  GeM + CBA      | 92.2  | 89.2 |
|  GeM + GCA_3    | 92.5  | 89.6 |
|  GeM + GCA_32   | 92.3  | 88.8 |
|  GeM + GCA_64   | 92.5  | 90.4 |
|  GeM + GCA_128  | 91.4  | 89.5 |


2.Super-Alignment
|     Training Sets                     | Pitts Test | MSLS Val  | SF-XL Val|
|      ----                             | ----       | ----      | ----     |
|  GSV-Cities                           | 92.5       | 90.4      | 92.7     |
|  + Pitts-30k                          | 93.1       | 90.3      | 93.0     |
|  + MSLS                               | 92.8       | 92.7      | 94.1     |
|  + SF-XL                              | 92.5       | 90.0      | 93.6     |
|  + Pitts-30k + MSLS + SF-XL           | 92.6       | 92.7      | 94.5     |


3.Embodied Re-ranking
| Setups                 | Pitts Test | MSLS Val |
| ----                   | ----       | -----    |
| baseline 1             | 92.6       | 92.7     |
| 1 + Embodied Re-ranking| 92.9       | 93.0     |
| baseline 2             | 93.1       | 90.3     |
| 2 + Embodied Re-ranking| 93.6       | 90.8     |