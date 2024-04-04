import torch
from torch import nn
import models.loralib as lora
from models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


BACKBONE = {
    "dinov2_vits14": vit_small,
    "dinov2_vitb14": vit_base,
    "dinov2_vitl14": vit_large,
    "dinov2_vitg14": vit_giant2,
}

CHANNELS_NUM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

class DINOv2(nn.Module):

    def __init__(self, backbone : str, num_trainable_blocks = 4, norm_layer = True, return_token = True):
        super().__init__()
        self.model = BACKBONE[backbone](patch_size = 14, img_size = 518, init_values = 1, block_chunks = 0)
        self.channels_num = CHANNELS_NUM[backbone]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        pretrained_model_path = f"/home/ubuntu/.cache/torch/hub/checkpoints/{backbone}_pretrain.pth"

        if pretrained_model_path:
            model_state_dict = torch.load(pretrained_model_path)
            self.model.load_state_dict(model_state_dict, strict = False)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        num_blocks = len(self.model.blocks)
        for i, block in enumerate(self.model.blocks):
            if i >= num_blocks - self.num_trainable_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        # lora.utils.mark_only_lora_as_trainable(self.model)


    def forward(self, x):

        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        for blk in self.model.blocks:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        f = f.reshape((B, H // 14, W // 14, self.channels_num)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f