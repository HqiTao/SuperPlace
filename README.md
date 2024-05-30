# CS-VGL

Cross-domain Switch-aware Re-parameterization for Visual Geo-Localization

## TODO

* add other datasets, like sf-xl, pittsburgh, tokyo (and their night versions) **OK**
* add the gradient computation feature **OK**
* add the adaptive lora or dora **OK**
* add a classification layer for domain recognition
* add auto-switch feature in testing

# Experiments                                       Pitts   Nordland     MSLS     Amstertime
0. SALAD                                            92.4    89.7                  60.8
1. use sf-xl dataset, no mixup features,            92.0    82.0(82.9)            56.4
2. 322 fine-tune based on 1                         92.4    
3. use sf-xl dataset, mixup features                92.0
4. 322 fine-tune based on 3                         92.3
5. 322 fine-tune based on 1(MSLS Val)               92.4    81.9         91.4     56.5
