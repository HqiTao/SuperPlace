from torch import nn

from models import dinov2_network
import models.aggregations as aggregations


class VGLNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = dinov2_network.DINOv2(backbone=args.backbone,
                               trainable_layers=args.trainable_layers)
        
        self.aggregation = get_aggregation(args)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x
    

def get_aggregation(args):
    if args.aggregation == "salad":
        return aggregations.SALAD(num_channels = dinov2_network.CHANNELS_NUM[args.backbone])
    elif args.aggregation == "netvlad":
        return aggregations.NetVLAD(dim=dinov2_network.CHANNELS_NUM[args.backbone], work_with_tokens=True)
    elif args.aggregation == "cosgem":
        return aggregations.CosGeM(features_dim=dinov2_network.CHANNELS_NUM[args.backbone], fc_output_dim=args.features_dim)
    elif args.aggregation == "kernelgem":
        return aggregations.KernelGeM(features_dim=dinov2_network.CHANNELS_NUM[args.backbone], fc_output_dim=args.features_dim)
    elif args.aggregation == "maskedchannelgem":
        return aggregations.MaskedChannelGeM(features_dim=dinov2_network.CHANNELS_NUM[args.backbone], kernel = 1)