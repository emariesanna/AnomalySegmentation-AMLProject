import torch
import torch.nn as nn
import numpy as np
import math
from utilities.state_dictionary import load_my_state_dict, load_my_quant_erfnet_state_dict
from eval.erfnet import ERFNet
import pandas as pd

def compute_output_dim(input_size, kernel_size, stride, padding, dilation=1, output_padding = 1):

    return math.floor((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

def count_flops_iops_and_params_full(model, input_tensor):
    model.eval()

    flops_per_layer = []
    iops_per_layer = []
    params_per_layer = []
    zero_params_per_layer = []
    layer_names = []
    layer_types = []

    x = input_tensor
    for name, module in model.named_modules():
        if name == "":
            continue  # skip the top-level model itself

        layer_names.append(name)
        layer_types.append(module.__class__.__name__)
        flops = 0
        iops = 0

        # Get weights if available
        w = getattr(module, "weight", None)
        weights = w() if callable(w) else w

        total_params = 0
        zero_params = 0
        non_zero_params = 0

        if weights is not None and isinstance(weights, torch.Tensor):
            try:
                if weights.is_quantized:
                    try:
                        real_weights = weights.dequantize()

                        # Verifica che il tipo sia float
                        if not torch.is_floating_point(real_weights):
                            raise ValueError("Il tensore dequantizzato non è a virgola mobile.")

                        #  Verifica che non sia ancora quantizzato
                        if real_weights.is_quantized:
                            raise ValueError("Il tensore risulta ancora quantizzato dopo la dequantizzazione.")

                        # Controlla valori sospetti
                        if torch.isnan(real_weights).any() or torch.isinf(real_weights).any():
                            raise ValueError("Valori NaN o inf trovati dopo la dequantizzazione.")

                    except Exception as e:
                        print(f"Errore durante la dequantizzazione dei pesi: {e}")
                    #real_weights = None
                else:
                    real_weights = weights

                total_params = real_weights.numel()
                zero_params = (real_weights == 0).sum().item()
                non_zero_params = total_params - zero_params

            except Exception as e:
                print(f"Errore nel calcolo dei pesi in {name}: {e}")

        try:
            if isinstance(module, (nn.Conv2d, nn.quantized.Conv2d, nn.intrinsic.quantized.ConvReLU2d)):
                # if isinstance(module, nn.Conv2d):
                #     print(f"Layer {name} is a {module.__class__.__name__}")
                # else:
                #     print(f"Layer {name} is a quantized {module.__class__.__name__}")
                C_in = module.in_channels
                C_out = module.out_channels
                Kh, Kw = module.kernel_size
                Sh, Sw = module.stride
                Ph, Pw = module.padding
                Dh, Dw = module.dilation
                G = module.groups

                H_in, W_in = x.shape[2], x.shape[3]
                H_out = compute_output_dim(H_in, Kh, Sh, Ph, Dh)
                W_out = compute_output_dim(W_in, Kw, Sw, Pw, Dw)
                ops = C_out * H_out * W_out * (C_in // G) *Kh * Kw

                # Distribuisci FLOPs e IOPS in base a pesi zero/non zero
                if "quantized" in str(type(module)):
                    # per layer quantizzati consideriamo tutte le ops come IOPS
                    iops = 2 * ops
                    flops = 0
                else:
                    # print(f"Non-zero params: {non_zero_params}, Zero params: {zero_params}, Total params: {total_params}")
                    flops = 2 * ops * (non_zero_params / total_params) if total_params > 0 else 0
                    iops = 2 * ops * (zero_params / total_params) if total_params > 0 else 0 

                x = torch.ones((1, C_out, H_out, W_out))

                #x = torch.zeros((1, C_out, H_out, W_out))
            elif isinstance(module, (nn.ConvTranspose2d, nn.quantized.ConvTranspose2d)):
                # if isinstance(module, nn.quantized.ConvTranspose2d):
                #     print(f"Layer {name} is a quantized {module.__class__.__name__}")
                # else:
                #     print(f"Layer {name} is a {module.__class__.__name__}")
                C_in = module.in_channels
                C_out = module.out_channels
                Kh, Kw = module.kernel_size
                Sh, Sw = module.stride
                Ph, Pw = module.padding
                Dh, Dw = module.dilation
                G = module.groups
                output_padding_h, output_padding_w = module.output_padding if module.output_padding is not None else (0, 0)

                H_in, W_in = x.shape[2], x.shape[3]
                
                ops = C_out * H_in * W_in * (C_in // G) *Kh * Kw

                H_out = math.floor((H_in - 1) * Sh - 2 * Ph + Dh * (Kh - 1) + output_padding_h + 1)
                W_out = math.floor((W_in - 1) * Sw - 2 * Pw + Dw * (Kw - 1) + output_padding_w + 1)

                # Distribuisci FLOPs e IOPS in base a pesi zero/non zero
                if "quantized" in str(type(module)):
                    # per layer quantizzati consideriamo tutte le ops come IOPS
                    iops = 2 * ops
                    flops = 0
                else:
                    # print(f"Non-zero params: {non_zero_params}, Zero params: {zero_params}, Total params: {total_params}")
                    flops = 2 * ops * (non_zero_params / total_params) if total_params > 0 else 2 * ops
                    iops = 2 * ops * (zero_params / total_params) if total_params > 0 else 0
                # print(f"Layer {name}: output shape {(H_out,W_out)}, kernel {( Kh, Kw)}, in_channels {C_in}, out_channels {C_out}")
                flops =0
                iops = 0
                x = torch.ones((1, C_out, H_out, W_out))


            elif isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                ops = in_features * out_features

                flops = 2 * ops * (non_zero_params / total_params) if total_params > 0 else 2 * ops
                iops = 2 * ops * (zero_params / total_params) if total_params > 0 else 0


                flops =0
                iops = 0

                x = torch.ones((1, out_features))

            elif isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.AdaptiveAvgPool2d)):
                # Questi layer fanno operazioni per elemento, senza pesi
                flops = x.numel()
                iops = 0

            elif isinstance(module, nn.MaxPool2d):
                Kh, Kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
                Sh, Sw = module.stride if module.stride is not None else (1, 1)
                Ph, Pw = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                H_in, W_in = x.shape[2], x.shape[3]
                H_out = compute_output_dim(H_in, Kh, Sh, Ph)
                W_out = compute_output_dim(W_in, Kw, Sw, Pw)
                flops = x.shape[1] * H_out * W_out * Kh * Kw
                iops = 0

                flops =0
                iops = 0
                x= torch.ones((1, x.shape[1], H_out, W_out))
                #x = torch.zeros((1, x.shape[1], H_out, W_out))"""

            else:
                # Altri layer considerati senza FLOPs
                #print(f"Skipping FLOP estimation for layer {name} due to unsupported type: {type(module)}")
                flops = 0
                iops = 0

        except Exception as e:
            pass#print(f"Skipping FLOP estimation for layer {name} due to error: {e}")

        flops_per_layer.append(flops)
        iops_per_layer.append(iops)
        params_per_layer.append(total_params)
        zero_params_per_layer.append(zero_params)

    return {
        "layer_names": layer_names,
        "layer_types": layer_types,
        "flops_per_layer": flops_per_layer,
        "iops_per_layer": iops_per_layer,
        "params_per_layer": params_per_layer,
        "zero_params_per_layer": zero_params_per_layer,
        "total_flops": int(np.sum(flops_per_layer)),
        "total_iops": int(np.sum(iops_per_layer)),
        "total_params": int(np.sum(params_per_layer)),
        "total_zero_params": int(np.sum(zero_params_per_layer)),
    }

def print_info(info_o):
    for i in range(len(info_o["layer_names"])):
            def format_num(n):
                return f"{n:_}"

            print(f"{info_o['layer_names'][i]} ({info_o['layer_types'][i]}): "
                f"FLOPs={format_num(info_o['flops_per_layer'][i])} "
                f"IOPS={format_num(info_o['iops_per_layer'][i])} "
                f"Params={format_num(info_o['params_per_layer'][i])} "
                f"Zeri={format_num(info_o['zero_params_per_layer'][i])}")


if __name__ == "__main__":

    device= 'cpu'  # Use 'cuda' if you have a GPU available


    pruned_model = ERFNet(num_classes=20)
    pruned_model.load_state_dict(torch.load('./trained_models/pruning_quantization/colab/erfnet_finetuned_pruned_30%.pth',map_location=torch.device('cpu')))
    pruned_model = pruned_model.to(device)  # Assign the pruned model to the correct variable

    Original_model = ERFNet(num_classes=20)
    Original_model= load_my_state_dict(Original_model,torch.load('./trained_models/finetuning/erfnet_finetuned.pth',map_location=torch.device('cpu')))
    Original_model = Original_model.to(device) # Assign the original model to the correct variable


    # Load the quantized model using the quantized model loading function
    #quantized_model = QuantizableERFNet(num_classes=20)
    #quantized_model = load_my_quant_state_dict(quantized_model,'quantized_erfnet.pth',d = 'cpu')
    #quantized_model = quantized_model.to(device) # Move to CUDA after correct initialization
    #loaded_model.to(device)
    loaded_quantized_model = ERFNet(num_classes=20)
    loaded_quantized_model = load_my_quant_erfnet_state_dict('./trained_models/pruning_quantization/colab/erfnet_finetuned_pruned_30%_quantized_fx.pth')
    loaded_quantized_model.to('cpu')

    """# Force all parameters and buffers to be on the CUDA device
    for name, param in quantized_model.named_parameters():
        param.data = param.data.to(device) # Ensure all parameters are on CUDA
    for name, buffer in quantized_model.named_buffers():
        buffer.data = buffer.data.to(device) # Ensure all buffers are on CUDA"""

    loaded_quantized_nopruned_model = ERFNet(num_classes=20)
    loaded_quantized_nopruned_model = load_my_quant_erfnet_state_dict('./trained_models/pruning_quantization/colab/erfnet_finetuned_quantized_fx.pth')
    loaded_quantized_nopruned_model.to('cpu')

    inputs =  torch.ones(1, 3, 512, 1024)  # Example input tensor, adjust size as needed

    # modello quantizzato con FX Graph Mode

    def format_num(n):
        return f"{n:_}"

    print("\n\nPruned and Quantized Model:")
    info_pq = count_flops_iops_and_params_full(loaded_quantized_model, inputs.to('cpu'))
    #print_info(info_pq)

    print(f"Total flops/iops for pruned and quantized model: {format_num(info_pq['total_flops'])}/{format_num(np.sum(info_pq['iops_per_layer']))}")
    print(f"Total params/zeros for pruned and quantized model: {format_num(np.sum(info_pq['params_per_layer']))}/{format_num(np.sum(info_pq['zero_params_per_layer']))}")


    print("\n\nQuantized Model:")
    info_q = count_flops_iops_and_params_full(loaded_quantized_nopruned_model, inputs.to('cpu'))
    #print_info(info_q)

    print(f"Total flops/iops for quantized model: {format_num(np.sum(info_q['flops_per_layer']))}/{format_num(np.sum(info_q['iops_per_layer']))}")
    print(f"Total params/zeros for quantized model: {format_num(np.sum(info_q['params_per_layer']))}/{format_num(np.sum(info_q['zero_params_per_layer']))}")


    print("\n\nPruned Model:")
    info_p = count_flops_iops_and_params_full(pruned_model, inputs.to('cpu'))
    #print_info(info_p)

    print(f"Total flops/iops for pruned model: {format_num(info_p['total_flops'])}/{format_num(np.sum(info_p['iops_per_layer']))}")
    print(f"Total params/zeros for pruned model: {format_num(np.sum(info_p['params_per_layer']))}/{format_num(np.sum(info_p['zero_params_per_layer']))}")


    print("\n\nOriginal Model:")
    info_o = count_flops_iops_and_params_full(Original_model, inputs.to('cpu'))
    #print_info(info_o)

    print(f"Total flops/iops for model: {format_num(np.sum(info_o['flops_per_layer']))}/{format_num(np.sum(info_o['iops_per_layer']))}")
    print(f"Total params/zeros for model: {format_num(np.sum(info_o['params_per_layer']))}/{format_num(np.sum(info_o['zero_params_per_layer']))}")



    """def info_to_dataframe(info, model_name):
        data = {
            "Layer": info["layer_names"],
            "Type": info["layer_types"],
            f"{model_name}_Params": info["params_per_layer"],
            f"{model_name}_Zeros": info["zero_params_per_layer"],
            f"{model_name}_FLOPs": info["flops_per_layer"],
            f"{model_name}_IOPs": info["iops_per_layer"],
        }
        return pd.DataFrame(data)

    df_o = info_to_dataframe(info_o, "Original")
    df_p = info_to_dataframe(info_p, "Pruned")
    df_q = info_to_dataframe(info_q, "Quantized")
    df_pq = info_to_dataframe(info_pq, "PrunedQuantized")

    # Merge all dataframes on Layer and Type
    df_merged = df_o.merge(df_p, on=["Layer", "Type"], how="outer") \
                    .merge(df_q, on=["Layer", "Type"], how="outer") \
                    .merge(df_pq, on=["Layer", "Type"], how="outer")

    # Fill NaNs with 0 for clarity
    df_merged = df_merged.fillna(0)

    # Format numbers with underscores for readability
    def fmt(x):
        return f"{int(x):_}"

    for col in df_merged.columns:
        if "Params" in col or "Zeros" in col or "FLOPs" in col or "IOPs" in col:
            df_merged[col] = df_merged[col].apply(fmt)

    # Print as a table
    print(df_merged.to_string(index=False))"""
    