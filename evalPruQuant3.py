from eval.erfnet import ERFNet
from fvcore.nn import FlopCountAnalysis

import torch

from utilities.state_dictionary import load_my_quant_erfnet_state_dict, load_my_state_dict

def get_flops(model):

    input = torch.randn(1, 3, 512, 1024)
    flops = FlopCountAnalysis(model, input)

    return flops


def main():

    device = 'cpu'

    Original_model = ERFNet(num_classes=20)
    # Original_model = load_my_state_dict(Original_model,torch.load('./trained_models/finetuning/erfnet_finetuned.pth',map_location=torch.device('cpu')))
    Original_model = Original_model.to(device)
    Original_model.eval()
    flops = get_flops(Original_model)
    print("Original model:")
    print(f"Total FLOPs: {flops.total()/1e9:.2f} GFLOPs")
    print(flops.by_operator())

if __name__ == '__main__':
    main()
