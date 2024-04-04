from torch import nn

from models.dinov2_network import DINOv2
import models.aggregations as aggregations


class VGLNet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = DINOv2(backbone=backbone)
        self.aggregation = get_aggregation("salad")
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x
    

def get_aggregation(args):
    if args == "salad":
        return aggregations.SALAD()