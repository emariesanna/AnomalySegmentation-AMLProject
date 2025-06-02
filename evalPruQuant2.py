import torch
from utilities.state_dictionary import load_my_state_dict, load_my_quant_erfnet_state_dict
from eval.erfnet import ERFNet
import torch.nn as nn
import math

def count_conv2d_flops(conv_module, input):

    N, C_in, H_in, W_in = input.shape
    C_out = conv_module.out_channels
    K_h, K_w = conv_module.kernel_size
    S_h, S_w = conv_module.stride
    P_h, P_w = conv_module.padding
    D_h, D_w = conv_module.dilation

    # Calcolo dimensioni output
    H_out = math.floor((H_in + 2 * P_h - D_h * (K_h - 1) - 1) / S_h + 1)
    W_out = math.floor((W_in + 2 * P_w - D_w * (K_w - 1) - 1) / S_w + 1)

    # MAC per singola posizione di output per canale
    mac_per_output_element = K_h * K_w * C_in

    # Totale MAC
    total_mac = N * H_out * W_out * C_out * mac_per_output_element

    # FLOPs (assumendo 2 FLOPs per MAC)
    total_flops = total_mac * 2

    output = conv_module(input)
    if output.shape[1] != C_out or output.shape[2] != H_out or output.shape[3] != W_out:
        raise ValueError("Output shape does not match expected dimensions.")

    return total_flops, output

def count_batchnorm2d_flops(bn_module, input):
    N, C, H, W = input.shape
    num_elements = N * C * H * W

    # 4 FLOPs per elemento: (x - mean) / std * gamma + beta
    total_flops = num_elements * 4

    output = bn_module(input)
    if output.shape != input.shape:
        raise ValueError("Output shape does not match input shape.")
    
    return total_flops, output


def count_convtranspose2d_flops(conv_module, input):
    N, C_in, H_in, W_in = input.shape
    C_out = conv_module.out_channels
    K_h, K_w = conv_module.kernel_size
    S_h, S_w = conv_module.stride
    P_h, P_w = conv_module.padding
    O_h, O_w = conv_module.output_padding
    D_h, D_w = conv_module.dilation

    # Calcolo dimensioni output secondo formula PyTorch (documentazione ufficiale)
    H_out = (H_in - 1) * S_h - 2 * P_h + D_h * (K_h - 1) + O_h + 1
    W_out = (W_in - 1) * S_w - 2 * P_w + D_w * (K_w - 1) + O_w + 1

    # MAC per singola posizione di output per canale
    mac_per_output_element = K_h * K_w * C_in

    # Totale MAC
    total_mac = N * H_out * W_out * C_out * mac_per_output_element

    # FLOPs (2 per MAC)
    total_flops = total_mac * 2

    # Forward e controllo output shape
    output = conv_module(input)
    if output.shape[1] != C_out or output.shape[2] != H_out or output.shape[3] != W_out:
        raise ValueError("Output shape does not match expected dimensions.")

    return total_flops, output



def get_ops_and_params(model):

    state_dict = model.state_dict()
    
    total_params = state_dict.values()
    total_params = sum(p.numel() for p in total_params if torch.is_tensor(p))
    nonzero_params = 0

    for name, param in state_dict.items():

        if torch.is_tensor(param):
            nonzero_params += (param !=0).sum().item()
            #nonzero_params += torch.count_nonzero(param).item()

    module_types = set()
    input = torch.ones(1, 3, 512, 1024)
    flops = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            print(f"Module: {name}, Type: {type(module)}")
            module_types.add(type(module))
            if isinstance(module, nn.Conv2d):
                op, input = count_conv2d_flops(module, input)
                flops += op
            elif isinstance(module, nn.BatchNorm2d):
                op, input = count_batchnorm2d_flops(module, input)
                flops += op
            elif isinstance(module, nn.ConvTranspose2d):
                op, input = count_convtranspose2d_flops(module, input)
                flops += op
            

    print("Module types encountered:")
    for t in module_types:
        print(t)
        
    return nonzero_params
            


def main():

    device = 'cpu'

    Original_model = ERFNet(num_classes=20)
    Original_model = load_my_state_dict(Original_model,torch.load('./trained_models/finetuning/erfnet_finetuned.pth',map_location=torch.device('cpu')))
    Original_model = Original_model.to(device)
    Original_model.eval()
    print("Original model:")
    params = get_ops_and_params(Original_model)
    print( f"Total parameters in original model: {params}")

    pruned_model = ERFNet(num_classes=20)
    pruned_model = load_my_state_dict(pruned_model, torch.load('./trained_models/pruning_quantization/colab/erfnet_finetuned_pruned_30%.pth',map_location=torch.device('cpu')))
    pruned_model = pruned_model.to(device)
    pruned_model.eval()
    print("Pruned model:")
    params = get_ops_and_params(pruned_model)
    print( f"Total parameters in pruned model: {params}")

    quantized_model = load_my_quant_erfnet_state_dict('./trained_models/pruning_quantization/colab/erfnet_finetuned_quantized_fx.pth')
    quantized_model = quantized_model.to(device)
    quantized_model.eval()
    print("Quantized model:")
    params = get_ops_and_params(quantized_model)
    print( f"Total parameters in quantized model: {params}")

    quantized_pruned_model = load_my_quant_erfnet_state_dict('./trained_models/pruning_quantization/colab/erfnet_finetuned_pruned_30%_quantized_fx.pth')
    quantized_pruned_model = quantized_pruned_model.to(device)
    quantized_pruned_model.eval()
    print("Quantized pruned model:")
    params = get_ops_and_params(quantized_pruned_model)
    print( f"Total parameters in pruned and quantized model: {params}")


if __name__ == '__main__':
    main()



