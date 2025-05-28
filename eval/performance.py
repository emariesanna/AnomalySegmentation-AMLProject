import torch.nn as nn


def count_params(model):

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantized_params = sum(b.numel() for _, b in model.named_buffers() if b.requires_grad)
    total_params += quantized_params

    return total_params

def count_active_parameters(model):
    total_params = 0
    active_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()  # Numero totale di parametri
            active_params += (param != 0).sum().item()  # Conta i pesi non azzerati
    return total_params, active_params

def get_model_size(model):
    """ Restituisce la dimensione del modello in MB. """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size  # Byte totali
    return total_size / (1024 ** 2)  # Conversione in MB

# Funzione per calcolare i FLOPs del modello
def count_flops(model, input_tensor):
    flops = 0
    def count_layer_flops(layer, input, output):
        nonlocal flops
        # Conv2d layer
        if isinstance(layer, nn.Conv2d):
            in_channels = input[0].shape[1]
            out_channels = output.shape[1]
            kernel_size = layer.kernel_size[0]
            height, width = input[0].shape[2], input[0].shape[3]
            flops += in_channels * out_channels * kernel_size * kernel_size * height * width
        
        # ConvTranspose2d layer
        elif isinstance(layer, nn.ConvTranspose2d):
            in_channels = input[0].shape[1]
            out_channels = output.shape[1]
            kernel_size = layer.kernel_size[0]
            height, width = input[0].shape[2], input[0].shape[3]
            flops += in_channels * out_channels * kernel_size * kernel_size * height * width
        
        # Linear layers (Fully connected)
        elif isinstance(layer, nn.Linear):
            input_features = input[0].shape[1]
            output_features = output.shape[1]
            flops += input_features * output_features
        
    # Hook function to count FLOPs
    hooks = []
    for layer in model.modules():
        hook = layer.register_forward_hook(count_layer_flops)
        hooks.append(hook)
    
    # Dummy forward pass to calculate FLOPs
    model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

    return flops