from eval.eval_iou import main as eval_iou
import torch
from utilities.state_dictionary import load_my_state_dict, load_my_quant_erfnet_state_dict
from models.erfnet import ERFNet

def main():


    # print("ERFNet pruned and quantized.")
    # weightspath = "trained_models/pruning_quantization/colab/erfnet_finetuned_pruned_30%_quantized_fx.pth"
    # model = load_my_quant_erfnet_state_dict(weightspath, device='cpu', print_model=False)
    # model.eval()
    # model.cpu()
    # eval_iou(model, datadir='./Dataset/Cityscapes', cpu=True, num_classes=20, batch_size=10, ignoreIndex=19, print_images=0, imagesize=(512, 1024), split=0)

    print("ERFNet quantized.")
    weightspath = "trained_models/pruning_quantization/erfnet/erfnet_quantized.pth"
    model = load_my_quant_erfnet_state_dict(weightspath, device='cpu', print_model=False)
    model.eval()
    model.cpu()
    eval_iou(model, datadir='./Dataset/Cityscapes', cpu=True, num_classes=20, batch_size=10, ignoreIndex=19, print_images=0, imagesize=(512, 1024), split=0)
    """

    model = ERFNet(num_classes=20)
    print("ERFNet pruned.")
    weightspath = "trained_models/pruning_quantization/colab/erfnet_finetuned_pruned_30%.pth"
    model = load_my_state_dict(model, torch.load(weightspath, map_location=torch.device('cpu')))
    model.eval()
    model.cpu()
    eval_iou(model, datadir='./Dataset/Cityscapes', cpu=True, num_classes=20, batch_size=10, ignoreIndex=19, print_images=0, imagesize=(512, 1024), split=10)

    model = ERFNet(num_classes=20)
    print("ERFNet original.")
    weightspath = "trained_models/finetuning/erfnet_finetuned.pth"
    model = load_my_state_dict(model, torch.load(weightspath, map_location=torch.device('cpu')))
    model.eval()
    model.cpu()
    eval_iou(model, datadir='./Dataset/Cityscapes', cpu=True, num_classes=20, batch_size=10, ignoreIndex=19, print_images=0, imagesize=(512, 1024), split=10)
    """

if __name__ == '__main__':
    main()
