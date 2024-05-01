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
        return aggregations.NetVLAD(dim=dinov2_network.CHANNELS_NUM[args.backbone], normalize_input=False, work_with_tokens=True)
    elif args.aggregation == "cosgem":
        return aggregations.CosGeM(features_dim=dinov2_network.CHANNELS_NUM[args.backbone], fc_output_dim=args.features_dim)
    elif args.aggregation == "cagem":
        return aggregations.CAGeM(features_dim=dinov2_network.CHANNELS_NUM[args.backbone], channels_num = args.channels_num)
    elif args.aggregation == "cls":
        return aggregations.CLS()
    elif args.aggregation == "mixedgem":
        return aggregations.MixedGeM(num_channels = dinov2_network.CHANNELS_NUM[args.backbone], 
                                     num_hiddens = None, 
                                     num_clusters = 192, 
                                     dim_clusters = 2)