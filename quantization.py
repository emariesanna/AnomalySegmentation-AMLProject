import sys
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quant
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
from eval.eval_iou import main as eval_iou
from utilities.state_dictionary import load_my_quant_fx_state_dict, load_my_state_dict
from eval.dataset import get_cityscapes_loader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

MODEL_NAME = 'ENet'

PRUNED = False  # Set to True if you want to use the pruned model


def calibrate(model, data_loader):
        step = 0
        model.eval()
        with torch.no_grad():
            for image, target, _, _ in data_loader:
                model(image)

                step += 1
                print(f"Step {step}/{len(data_loader)}")


def main():

    if MODEL_NAME == 'ERFNet':
        from eval.erfnet import ERFNet
        model = ERFNet(num_classes=20)
    elif MODEL_NAME == 'ENet':
        from eval.enet import ENet
        model = ENet(num_classes=20)
    elif MODEL_NAME == 'BiSeNet':
        from eval.bisenet import BiSeNetV1
        model = BiSeNetV1(num_classes=20)

    if PRUNED:
        weightspath = f"./trained_models/pruning_quantization/{MODEL_NAME.lower()}/{MODEL_NAME.lower()}_pruned.pth"
    else:
        weightspath = f"./trained_models/finetuning/{MODEL_NAME.lower()}_finetuned.pth"

    model_to_quantize = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))



    model_to_quantize.to('cpu')


    dataloader = get_cityscapes_loader("./Dataset/Cityscapes", 1, 'val', num_workers=4,imagesize = (512, 1024))

    model_to_quantize.eval()

    qconfig_opt = get_default_qconfig("x86")

    qconfig_mapping = QConfigMapping().set_global(qconfig_opt).set_object_type(
                                      torch.nn.ConvTranspose2d, get_default_qconfig("qnnpack")
                                  )  # qconfig_opt is an optional qconfig, either a valid qconfig or None
    example_inputs = dataloader.dataset[0][0].unsqueeze(0)
    prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)

    calibrate(prepared_model, dataloader)
    quantized_model = convert_fx(prepared_model)

    
    fx_graph_mode_model_file_path = f"./trained_models/pruning_quantization/{MODEL_NAME.lower()}/{MODEL_NAME.lower()}_quantized.pth"

    #torch.save(quantized_model, fx_graph_mode_model_file_path)
    torch.save(quantized_model.state_dict(),  fx_graph_mode_model_file_path)

    if MODEL_NAME == 'ERFNet':
        model = ERFNet(num_classes=20)
    elif MODEL_NAME == 'ENet':
        model = ENet(num_classes=20)
    elif MODEL_NAME == 'BiSeNet':
        model = BiSeNetV1(num_classes=20)

    loaded_quantized_model = load_my_quant_fx_state_dict(model, fx_graph_mode_model_file_path, printing=False)

    iou = eval_iou(loaded_quantized_model, "./Dataset/Cityscapes", cpu=False, num_classes=20, ignoreIndex=19)


if __name__ == "__main__":
    main()