from torch import nn

from models.dinov2_network import DINOv2
import models.aggregations as aggregations


class VPRNet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = DINOv2(backbone=backbone)
        self.aggregation = get_aggregation("netvlad")
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x
    

def get_aggregation(args):
    if args == "gem":
        return aggregations.GeM(work_with_tokens=args.work_with_tokens)
    elif args == "salad":
        return aggregations.SALAD()
    elif args == "netvlad":
        return aggregations.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)