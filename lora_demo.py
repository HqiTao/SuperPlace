import models.loralib as lora

from models.vpr_network import VPRNet
from utils.printer import print_trainable_parameters   


if __name__ == "__main__":

    model = VPRNet(backbone="dinov2_vitb14")
    model = model.to("cuda")

    # lora.mark_only_lora_as_trainable(model)
    print_trainable_parameters(model)