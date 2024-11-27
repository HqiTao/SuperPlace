from torch import nn

from models import dinov2_network
import models.aggregations as aggregations

import torchvision.transforms as transforms
import time

import torch
import logging
import torchvision
from typing import Tuple

import clip  # assuming you have the CLIP library installed

from transformers import ViTModel


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        # Load pre-trained CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Freeze all layers by default
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 4 layers of the vision encoder
        total_layers = len(self.clip_model.visual.transformer.resblocks)
        for i in range(total_layers - 4, total_layers):  # Unfreeze layers 8 to 11
            for param in self.clip_model.visual.transformer.resblocks[i].parameters():
                param.requires_grad = True

            
    def forward(self, x):
        # Get features from the CLIP vision encoder (image input)
        
        B, _, _,_ = x.shape

        x = self.clip_model.visual.conv1(x)  # shape = [batch_size, width, num_patches, num_patches]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [batch_size, width, num_patches^2]
        x = x.permute(0, 2, 1)  # shape = [batch_size, num_patches^2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [batch_size, num_patches^2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        # Pass through the transformer layers
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.visual.transformer(x)  # shape = [num_patches^2 + 1, batch_size, width]
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Here x contains the feature map for each patch, including the class token.
        image_features = x[:, 1:, :]
        # print(image_features.shape)
        image_features = image_features.reshape((B, 7, 7, 768)).permute(0, 3, 1, 2)
        
        return image_features
        # return image_features, x[:, 0, :] # for SALAD

class VGLNet_CLIP(nn.Module):

    def __init__(self, args):
        super().__init__()


        self.backbone = CLIP()
        # Aggregation network or task-specific layers (if necessary)
        self.aggregation = get_aggregation(args, channels= 768, fc_output_dim = 768)
        
    def forward(self, x):
        
        image_features = self.backbone(x)
        x = self.aggregation(image_features)

        return x

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 1024,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "EfficientNet_B0": 1280,
    "EfficientNet_B1": 1280,
    "EfficientNet_B2": 1408,
    "EfficientNet_B3": 1536,
    "EfficientNet_B4": 1792,
    "EfficientNet_B5": 2048,
    "EfficientNet_B6": 2304,
    "EfficientNet_B7": 2560,
    "ViT":768,
}
    
class VGLNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(backbone=args.backbone,
                               trainable_layers=args.trainable_layers,
                               return_token = False)
        
        self.aggregation = get_aggregation(args, channels=dinov2_network.CHANNELS_NUM[args.backbone], fc_output_dim = dinov2_network.CHANNELS_NUM[args.backbone])
        
    def forward(self, x):

        x = self.backbone(x)
        x = self.aggregation(x)
        return x
    
class VGLNet_Test(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(backbone=args.backbone,
                               trainable_layers=args.trainable_layers,
                                return_token = False)
        
        self.aggregation = get_aggregation(args, channels=dinov2_network.CHANNELS_NUM[args.backbone], fc_output_dim = dinov2_network.CHANNELS_NUM[args.backbone])
        
        self.all_time = 0
        
    def forward(self, x):

        if not self.training:
            b, c, h, w = x.shape
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            x = transforms.functional.resize(x, [h, w], antialias=True)

        x = self.backbone(x)
        x = self.aggregation(x)

        return x
    
class GeoLocalizationNet(nn.Module):
    def __init__(self, args, backbone : str, fc_output_dim : int, train_all_layers : bool = False):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
            train_all_layers (bool): whether to freeze the first layers of the backbone during training or not.
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone, train_all_layers)
        self.aggregation = get_aggregation(args, channels=features_dim, fc_output_dim = features_dim)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x
    

def get_aggregation(args, channels = None , fc_output_dim = None):
    if args.aggregation == "salad":
        return aggregations.SALAD(num_channels = channels)
    elif args.aggregation == "netvlad":
        return aggregations.NetVLAD(clusters_num=args.clusters, dim=channels, work_with_tokens=args.use_cls, linear_dim = args.linear_dim, work_with_linear = args.use_linear)
    elif args.aggregation == "cosgem":
        return aggregations.CosGeM(features_dim= channels, fc_output_dim=args.features_dim)
    elif args.aggregation == "cls":
        return aggregations.CLS()
    elif args.aggregation == "g2m":
        return aggregations.G2M(
            # num_channels=640,
            num_channels=channels,
            fc_output_dim=fc_output_dim,
            num_hiddens=args.num_hiddens,
            use_cls=args.use_cls,
            use_ca=args.use_ca,
            pooling_method=args.ca_method,
        )


def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone_name : str, train_all_layers : bool) -> Tuple[torch.nn.Module, int]:
    backbone = get_pretrained_torchvision_model(backbone_name)
    if backbone_name.startswith("ResNet"):
        if train_all_layers:
            logging.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")

        layers = list(backbone.children())[:-3]  # Remove avg pooling and FC layer
    
    elif backbone_name == "VGG16":
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        if train_all_layers:
            logging.debug("Train all layers of the VGG-16")
        else:
            for layer in layers[:-5]:
                for p in layer.parameters():
                    p.requires_grad = False
            logging.debug("Train last layers of the VGG-16, freeze the previous ones")

    elif backbone_name.startswith("EfficientNet"):
        if train_all_layers:
            logging.debug(f"Train all layers of the {backbone_name}")
        else:
            for name, child in backbone.features.named_children():
                if name == "5": # Freeze layers before block 5
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only the last three blocks of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2] # Remove avg pooling and FC layer
        
    elif backbone_name.startswith("vit"):
        backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        logging.debug(f"Freeze all the layers up to tranformer encoder {8}")
        for p in backbone.parameters():
            p.requires_grad = False
        for name, child in backbone.encoder.layer.named_children():
            if int(name) > 8:
                for params in child.parameters():
                    params.requires_grad = True
        
        return backbone, 768
    
    backbone = torch.nn.Sequential(*layers)
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim