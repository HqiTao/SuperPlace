import logging

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")


def print_trainable_layers(model):
    """
    Prints the name of trainable parameters in the model.
    """
    layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad and "bias" not in name and "aggregation" not in name:
            layer_names.append(".".join(name.split(".")[2:-1]))

    logging.info(", ".join(layer_names))

