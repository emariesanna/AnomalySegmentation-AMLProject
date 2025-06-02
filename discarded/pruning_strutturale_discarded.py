
from argparse import ArgumentParser
from pruning_quantization.erfnet import ERFNet, DownsamplerBlock, non_bottleneck_1d, UpsamplerBlock
import torch
import torch.nn as nn
import os

from utilities.state_dictionary import load_my_state_dict
from eval.dataset import get_cityscapes_loader
from tqdm import tqdm
from eval.eval_iou import main as eval_iou
from eval.evalAnomaly import main as evalAnomaly
from train.finetuning import train


DatasetDir = {
    "LostFound": "./Dataset/FS_LostFound_full/images/*.png",
    "FSstatic": "./Dataset/fs_static/images/*.jpg",
    "RoadAnomaly": "./Dataset/RoadAnomaly/images/*.jpg",
    "RoadAnomaly21": "./Dataset/RoadAnomaly21/images/*.png",
    "RoadObstacle21": "./Dataset/RoadObstacle21/images/*.webp",
              }


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--datadir', default="./Dataset/Cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--steps-loss', type=int, default=1)
    parser.add_argument('--savedir', default="./trained_models")
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def eval_anomaly(model):
    for dataset in DatasetDir.keys():
        print("Dataset " + dataset)
        dataset_dir = DatasetDir[dataset]
        prc_auc, fpr = evalAnomaly(dataset_dir, model, "VoidClass", cpu=False, imagesize=(512, 1024))
        print(f'AUPRC score: {prc_auc*100.0:.2f}\tFPR@TPR95: {fpr*100.0:.2f}%')


def prune_downsampler(old_block, in_channels, in_channels_mask, prune_ratio=0.3):

    old_conv_channels = old_block.conv.out_channels
    num_pruned = max(1, int(old_conv_channels * (1 - prune_ratio)))

    # Seleziona i filtri più importanti via L1 norm
    filtered_weights = old_block.conv.weight.data[:, in_channels_mask, :, :] # shape: [outC_old, inC_mask, kH, kW]
    l1_norm = filtered_weights.abs().sum(dim=(1, 2, 3))
    _, conv_mask = torch.topk(l1_norm, num_pruned)
    conv_mask = conv_mask.sort()[0]

    out_channels = num_pruned + in_channels

    # Costruisci nuova conv
    new_conv = nn.Conv2d(
        in_channels, num_pruned, kernel_size=3, stride=2, padding=1, bias=True)
    new_conv.weight.data = old_block.conv.weight.data[conv_mask][:, in_channels_mask, :, :].clone()
    new_conv.bias.data = old_block.conv.bias.data[conv_mask].clone()

    # BatchNorm: conv path + maxpool path
    new_bn = nn.BatchNorm2d(out_channels, eps=1e-3)
    new_bn.weight.data[:num_pruned] = old_block.bn.weight.data[conv_mask].clone()
    new_bn.bias.data[:num_pruned] = old_block.bn.bias.data[conv_mask].clone()
    new_bn.running_mean[:num_pruned] = old_block.bn.running_mean[conv_mask].clone()
    new_bn.running_var[:num_pruned] = old_block.bn.running_var[conv_mask].clone()

    maxpool_indices = in_channels_mask + old_conv_channels
    new_bn.weight.data[num_pruned:] = old_block.bn.weight.data[maxpool_indices].clone()
    new_bn.bias.data[num_pruned:] = old_block.bn.bias.data[maxpool_indices].clone()
    new_bn.running_mean[num_pruned:] = old_block.bn.running_mean[maxpool_indices].clone()
    new_bn.running_var[num_pruned:] = old_block.bn.running_var[maxpool_indices].clone()

    # Ricostruisci il blocco
    new_block = DownsamplerBlock(in_channels, out_channels)
    new_block.conv = new_conv
    new_block.bn = new_bn

    # Nuova maschera dei canali in output (conv + pool)
    out_channels_mask = torch.cat([conv_mask, maxpool_indices])

    return new_block, out_channels_mask



def prune_upsampler(old_block, in_channels, in_channels_mask, prune_ratio=0.3):

    old_conv_channels = old_block.conv.out_channels
    num_pruned = max(1, int(old_conv_channels * (1 - prune_ratio)))
 
    filtered_weights = old_block.conv.weight.data[in_channels_mask, :, :, :] # shape: [inC_mask, outC_old, k, k]
    l1_norm = filtered_weights.abs().sum(dim=(0, 2, 3))  # shape: (outC_old,)
    _, conv_mask = torch.topk(l1_norm, num_pruned)
    conv_mask = conv_mask.sort()[0] 

    new_conv = nn.ConvTranspose2d(
        in_channels, num_pruned,kernel_size=3,stride=2,padding=1,output_padding=1,bias=True)
    
    new_conv.weight.data = old_block.conv.weight.data[in_channels_mask][:, conv_mask, :, :].clone()
    new_conv.bias.data = old_block.conv.bias.data[conv_mask].clone()

    new_bn = nn.BatchNorm2d(num_pruned, eps=1e-3)
    new_bn.weight.data = old_block.bn.weight.data[conv_mask].clone()
    new_bn.bias.data = old_block.bn.bias.data[conv_mask].clone()
    new_bn.running_mean = old_block.bn.running_mean[conv_mask].clone()
    new_bn.running_var = old_block.bn.running_var[conv_mask].clone()

    new_block = UpsamplerBlock(in_channels, num_pruned)
    new_block.conv = new_conv
    new_block.bn = new_bn

    out_channels_mask = conv_mask

    return new_block, out_channels_mask



def prune_nb1d(old_block, in_channels, in_channels_mask):

    # conv3x1_1
    filtered_weights_conv3x1_1 = old_block.conv3x1_1.weight.data[:, in_channels_mask, :, :]
    l1_norm = filtered_weights_conv3x1_1.abs().sum(dim=(1, 2, 3))
    _, conv3x1_1_mask = torch.topk(l1_norm, in_channels)
    conv3x1_1_mask = conv3x1_1_mask.sort()[0]

    new_conv3x1_1 = nn.Conv2d(
        in_channels, in_channels, kernel_size=(3,1), stride=1, padding=(1,0), bias=True)
    new_conv3x1_1.weight.data = old_block.conv3x1_1.weight.data[conv3x1_1_mask][:, in_channels_mask, :, :].clone()
    new_conv3x1_1.bias.data = old_block.conv3x1_1.bias.data[conv3x1_1_mask].clone()

    # conv1x3_1
    filtered_weights_conv1x3_1 = old_block.conv1x3_1.weight.data[:, conv3x1_1_mask, :, :]
    l1_norm = filtered_weights_conv1x3_1.abs().sum(dim=(1, 2, 3))
    _, conv1x3_1_mask = torch.topk(l1_norm, in_channels)
    conv1x3_1_mask = conv1x3_1_mask.sort()[0]

    new_conv1x3_1 = nn.Conv2d(
        in_channels, in_channels, kernel_size=(1,3), stride=1, padding=(0,1), bias=True)
    new_conv1x3_1.weight.data = old_block.conv1x3_1.weight.data[conv1x3_1_mask][:, conv3x1_1_mask, :, :].clone()
    new_conv1x3_1.bias.data = old_block.conv1x3_1.bias.data[conv1x3_1_mask].clone()

    # bn1
    new_bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
    new_bn1.weight.data = old_block.bn1.weight.data[conv1x3_1_mask].clone()
    new_bn1.bias.data = old_block.bn1.bias.data[conv1x3_1_mask].clone()
    new_bn1.running_mean = old_block.bn1.running_mean[conv1x3_1_mask].clone()
    new_bn1.running_var = old_block.bn1.running_var[conv1x3_1_mask].clone()

    # conv3x1_2
    filtered_weights_conv3x1_2 = old_block.conv3x1_2.weight.data[:, conv1x3_1_mask, :, :]
    l1_norm = filtered_weights_conv3x1_2.abs().sum(dim=(1, 2, 3))
    _, conv3x1_2_mask = torch.topk(l1_norm, in_channels)
    conv3x1_2_mask = conv3x1_2_mask.sort()[0]

    new_conv3x1_2 = nn.Conv2d(
        in_channels, in_channels, kernel_size=(3,1), stride=1, padding=(1*old_block.dilated,0), 
        bias=True, dilation=(old_block.dilated, 1))
    new_conv3x1_2.weight.data = old_block.conv3x1_2.weight.data[conv3x1_2_mask][:, conv1x3_1_mask, :, :].clone()
    new_conv3x1_2.bias.data = old_block.conv3x1_2.bias.data[conv3x1_2_mask].clone()

    # conv1x3_2
    filtered_weights_conv1x3_2 = old_block.conv1x3_2.weight.data[:, conv3x1_2_mask, :, :]
    l1_norm = filtered_weights_conv1x3_2.abs().sum(dim=(1, 2, 3))
    _, conv1x3_2_mask = torch.topk(l1_norm, in_channels)
    conv1x3_2_mask = conv1x3_2_mask.sort()[0]

    new_conv1x3_2 = nn.Conv2d(
        in_channels, in_channels, kernel_size=(1,3), stride=1, padding=(0,1*old_block.dilated), 
        bias=True, dilation=(1, old_block.dilated))
    new_conv1x3_2.weight.data = old_block.conv1x3_2.weight.data[conv1x3_2_mask][:, conv3x1_2_mask, :, :].clone()
    new_conv1x3_2.bias.data = old_block.conv1x3_2.bias.data[conv1x3_2_mask].clone()

    # bn2
    new_bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)
    new_bn2.weight.data = old_block.bn2.weight.data[conv1x3_2_mask].clone()
    new_bn2.bias.data = old_block.bn2.bias.data[conv1x3_2_mask].clone()
    new_bn2.running_mean = old_block.bn2.running_mean[conv1x3_2_mask].clone()
    new_bn2.running_var = old_block.bn2.running_var[conv1x3_2_mask].clone()

    # dropout
    new_dropout = nn.Dropout2d(old_block.dropprob)

    new_block = non_bottleneck_1d(in_channels, old_block.dropprob, old_block.dilated)
    new_block.conv3x1_1 = new_conv3x1_1
    new_block.conv1x3_1 = new_conv1x3_1
    new_block.bn1 = new_bn1
    new_block.conv3x1_2 = new_conv3x1_2
    new_block.conv1x3_2 = new_conv1x3_2
    new_block.bn2 = new_bn2
    new_block.dropout = new_dropout

    out_channels_mask = conv1x3_2_mask

    return new_block, out_channels_mask



def adapt_last_conv(old_block, in_channels, in_channels_mask, num_classes):

    new_conv = nn.ConvTranspose2d(
        in_channels, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
    new_conv.weight.data = old_block.weight.data[in_channels_mask].clone()
    new_conv.bias.data = old_block.bias.data.clone()

    return new_conv



def main():
    model = ERFNet(num_classes=20)
    model = load_my_state_dict(model, torch.load("./trained_models/finetuning/erfnet_finetuned.pth", map_location="cpu"))

    model_path = "erfnet_model.pth"
    torch.save(model, model_path)
    file_size_mb = os.path.getsize(model_path) / (1024 ** 2)
    print(f"Model size: {file_size_mb:.2f} MB")

    current_mask = torch.arange(16)

    print("\nEncoder:")
    for idx, layer in enumerate(model.encoder.layers):
        if isinstance(layer, DownsamplerBlock):
            print(f"Pruning DownsamplerBlock @ {idx}")
            print(f"Input channels: {len(current_mask)}")

            old_out_channels = layer.conv.out_channels + layer.conv.in_channels
            pruned_layer, out_mask = prune_downsampler(layer, len(current_mask), current_mask)
            current_mask = out_mask.clone()
            model.encoder.layers[idx] = pruned_layer

            print(f"Channels before pruning: {old_out_channels}")
            print(f"Channels after pruning: {len(current_mask)}")

        elif isinstance(layer, non_bottleneck_1d):
            print(f"Pruning NonBottleneck1D @ {idx}")
            print(f"Input channels: {len(current_mask)}")

            old_out_channels = layer.conv1x3_2.out_channels
            pruned_layer, out_mask = prune_nb1d(layer, len(current_mask), current_mask)
            current_mask = out_mask.clone()
            model.encoder.layers[idx] = pruned_layer

            print(f"Channels before pruning: {old_out_channels}")
            print(f"Channels after pruning: {len(current_mask)}")

    print("\nDecoder:")
    for idx, layer in enumerate(model.decoder.layers):
        if isinstance(layer, UpsamplerBlock):
            print(f"Pruning UpsamplerBlock @ {idx}")
            print(f"Input channels: {len(current_mask)}")

            old_out_channels = layer.conv.out_channels
            pruned_layer, out_mask = prune_upsampler(layer, len(current_mask), current_mask)
            current_mask = out_mask.clone()
            model.decoder.layers[idx] = pruned_layer

            print(f"Channels before pruning: {old_out_channels}")
            print(f"Channels after pruning: {len(current_mask)}")

        elif isinstance(layer, non_bottleneck_1d):
            print(f"Pruning NonBottleneck1D @ {idx}")
            print(f"Input channels: {len(current_mask)}")

            old_out_channels = layer.conv1x3_2.out_channels
            pruned_layer, out_mask = prune_nb1d(layer, len(current_mask), current_mask)
            current_mask = out_mask.clone()
            model.decoder.layers[idx] = pruned_layer

            print(f"Channels before pruning: {old_out_channels}")
            print(f"Channels after pruning: {len(current_mask)}")

    print("\nAdapting last conv layer:")
    print(f"Input channels: {len(current_mask)}")
    model.decoder.output_conv = adapt_last_conv(model.decoder.output_conv, len(current_mask), current_mask, num_classes=20)

    pruned_model_path = "erfnet_pruned.pth"
    torch.save(model, pruned_model_path)
    file_size_mb = os.path.getsize(pruned_model_path) / (1024 ** 2)
    print(f"Pruned model size: {file_size_mb:.2f} MB")

    print("\nEvaluating pruned model:")

    eval_iou(model, "./Dataset/Cityscapes", cpu=False, num_classes=20, batch_size=10, ignoreIndex=-1, imagesize=(512, 1024))
    eval_anomaly(model)

    print("Training pruned model for 20 epochs:")
    model = train(get_args(), model, "erfnet")

    pruned_model_path = "erfnet_pruned_finetuned.pth"
    torch.save(model, pruned_model_path)
    file_size_mb = os.path.getsize(pruned_model_path) / (1024 ** 2)
    print(f"Pruned and finetuned model size: {file_size_mb:.2f} MB")

    eval_iou(model, "./Dataset/Cityscapes", cpu=False, num_classes=20, batch_size=10, ignoreIndex=-1, imagesize=(512, 1024))
    eval_anomaly(model)

if __name__ == "__main__":
    main()
