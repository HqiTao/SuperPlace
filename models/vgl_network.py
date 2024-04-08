from torch import nn

from models.dinov2_network import DINOv2
import models.aggregations as aggregations


class VGLNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.backbone = DINOv2(backbone=args.backbone,
                               num_trainable_blocks=args.num_trainable_blocks)
        
        self.aggregation = get_aggregation(args.aggregation)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x
    

def get_aggregation(aggregation_name):
    if aggregation_name == "salad":
        return aggregations.SALAD()