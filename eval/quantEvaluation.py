#calculate FLOPS, Memory usage, dimension,

import torch.quantization as quant
import torch.nn as nn
import torch.nn.utils.prune as prune

device = 'cpu'

pruned_model = ERFNet(num_classes=20)
pruned_model.load_state_dict(torch.load('/content/drive/MyDrive/trained_models/erfnet_finetuned_pruned_30%.pth',map_location=torch.device('cpu')))
pruned_model = pruned_model.to(device)  # Assign the pruned model to the correct variable

Original_model = ERFNet(num_classes=20)
Original_model= load_my_state_dict(Original_model,torch.load('/content/drive/MyDrive/trained_models/erfnet_finetuned.pth',map_location=torch.device('cpu')))
Original_model = Original_model.to(device) # Assign the original model to the correct variable


# Load the quantized model using the quantized model loading function
#quantized_model = QuantizableERFNet(num_classes=20)
#quantized_model = load_my_quant_state_dict(quantized_model,'quantized_erfnet.pth',d = 'cpu')
#quantized_model = quantized_model.to(device) # Move to CUDA after correct initialization
#loaded_model.to(device)
loaded_quantized_model = load_my_quant_fx_state_dict('/content/drive/MyDrive/trained_models/erfnet_finetuned_pruned_30%_quantized_fx.pth')
loaded_quantized_model.to('cpu')

#dimension
# Salva i modelli
torch.save(Original_model.state_dict(), 'EvalDimensionOriginal.pth')
torch.save(pruned_model.state_dict(), 'EvalDimensionPruned.pth')
torch.save(loaded_quantized_model.state_dict(), 'EvalDimensionFinal.pth')


# Misura le dimensioni
size_original = os.path.getsize('EvalDimensionOriginal.pth') / (1024 ** 2)  # Dimensioni in MB
size_pruned = os.path.getsize('EvalDimensionPruned.pth') / (1024 ** 2)
size_final = os.path.getsize('EvalDimensionFinal.pth') / (1024 ** 2)

print(f"ErfNet Size: {size_original:.2f} MB")
print(f"Pruned ErfNet Size: {size_pruned:.2f} MB")
print(f"Quantized ErfNet Size: {size_final:.2f} MB")


#Running  the iou of original and pruned models on cuda 
iou_P = eval_iou(pruned_model, datadir, cpu=False, num_classes=20, ignoreIndex=19)
iou_O = eval_iou(Original_model, datadir, cpu=False, num_classes=20, ignoreIndex=19)

#quantized model must run on cpu 
#i.e. it's slower because of cpu but actually very fast thinking that unquantized models are basically not runnable on cpu (more than 2 hours for eval_iou)
loaded_quantized_model.to('cpu')
iou_Q = eval_iou(loaded_quantized_model,datadir,cpu=True,num_classes=20,ignoreIndex = 19)

def count_active_parameters(model):
    total_params = 0
    active_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()  # Numero totale di parametri
            active_params += (param != 0).sum().item()  # Conta i pesi non azzerati
    return total_params, active_params


print('Original model: ')
total_o, active_o = count_active_parameters(Original_model)
print(f"Totale parametri: {total_o}")
print(f"Parametri attivi dopo il pruning: {active_o}  -> {active_o/(total_o+1) * 100}%")

print('Pruned model: ')
total_p, active_p = count_active_parameters(pruned_model)
print(f"Totale parametri: {total_p}")
print(f"Parametri attivi dopo il pruning: {active_p} -> {active_p/(total_p+1) * 100}%")

print('Quantized model: ')
total_q, active_q = count_active_parameters(loaded_quantized_model)
print(f"Totale parametri: {total_q}")
print(f"Parametri attivi dopo il pruning: {active_q}  -> {active_q/(total_q+1) * 100}%")


#!pip install fvcore

import time

def measure_inference_time(model, inputs,cpu = False):
    if cpu:
        model.to('cpu')
        inputs = inputs.to('cpu')
    else:
        model.to('cuda')
        inputs = inputs.to('cuda')

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        outputs = model(inputs)
    elapsed_time = time.time() - start_time
    return elapsed_time

#make sure you run loaded_quantized_model on cpu otherwise colab crashes
loaded_quantized_model.to('cpu')


inputs = torch.randn(1, 3, 512, 1024).to(device)  # Esempio di input


time_erfnet = measure_inference_time(Original_model, inputs)
print(f"ErfNet Inference Time: {time_erfnet:.4f}s")
time_pruned = measure_inference_time(pruned_model, inputs)
print(f"Pruned ErfNet Inference Time: {time_pruned:.4f}s")

#differences on cpu

time_erfnet = measure_inference_time(Original_model, inputs, cpu=True)
print(f"ErfNet Inference Time: {time_erfnet:.4f}s")
time_pruned = measure_inference_time(pruned_model, inputs, cpu=True)
print(f"Pruned ErfNet Inference Time: {time_pruned:.4f}s")

time_quantized = measure_inference_time(loaded_quantized_model, inputs, cpu=True)
print(f"Quantized ErfNet Inference Time: {time_quantized:.4f}s")


from fvcore.nn import FlopCountAnalysis

flops = FlopCountAnalysis(Original_model.to('cuda'), inputs)
print(f"ErfNet FLOPs: {flops.total()}")
flops_pruned = FlopCountAnalysis(pruned_model.to('cuda'), inputs)
print(f"Pruned ErfNet FLOPs: {flops_pruned.total()}")




# Move the loaded quantized model to CPU if it's not already
loaded_quantized_model.to('cpu')

# Create example inputs on CPU to match the model's device
inputs = torch.randn(1, 3, 512, 1024).to('cpu')

# Perform FlopCountAnalysis directly without modification on the CPU 
# This works if the model was originally quantized correctly and loaded correctly.
flops_quantized = FlopCountAnalysis(loaded_quantized_model, inputs)

# Print the total FLOPs
print(f"Quantized ErfNet FLOPs: {flops_quantized.total()}")
######errore qui