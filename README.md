# CS-VGL

Cross-domain Switch-aware Re-parameterization for Visual Geo-Localization

## TODO

* add other datasets, like sf-xl, pittsburgh, tokyo (and their night versions) **OK**
* add the gradient computation feature **OK**
* add the adaptive lora or dora **OK**
* add a classification layer for domain recognition
* add auto-switch feature in testing

## TODO
* MDR Loss + Triplet Loss
* KD Loss + Fast(Light, Mobile) ViT

# Re-produce
* change the pre-trained model path (dinov2_network.py, L37)


## Experiments

* DINO-L + SALAD + last 4 layers
* DINO-L + SALAD + Important layers
* DINO-L + SALAD + q, k, v, proj (LORA) value:
* DINO-L + SALAD + q, k, v, proj (DORA) value: 32
* DINO-L + SALAD + Important q, k, v, proj (DORA) value: 0.6, 0.5, 0.45, 0.55

## Experiments -extra datasets (SF-XL)

* DINO-B + SALAD + last 4 layers
* DINO-B + GeM + last 4 layers