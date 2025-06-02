import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp

import os
from utilities.state_dictionary import load_my_state_dict
from tqdm import tqdm
from eval.dataset import get_cityscapes_loader
from eval.erfnet import DownsamplerBlock

MODEL_NAMES = ["erfnet", "enet", "bisenet"]
IGNORED_LAYERS = {
    "enet": ["fullconv", "bottleneck50", "bottleneck51"],
    "erfnet": ["output_conv"],
    "bisenet": ["conv_out", "conv_out16", "conv_out32"],
}
PRUNING_AMOUNT = 0.3


def load_model(model_name):
    if model_name == "erfnet":
        from eval.erfnet import ERFNet
        weightspath = "./trained_models/finetuning/erfnet_finetuned.pth"
        model = ERFNet(num_classes=20)
        model = load_my_state_dict(model, torch.load(weightspath, map_location="cpu"))
    elif model_name == "enet":
        from eval.enet import ENet
        model = ENet(num_classes=20)
        weightspath = "./trained_models/finetuning/enet_finetuned.pth"
        model = load_my_state_dict(model, torch.load(weightspath, map_location="cpu"))
    elif model_name == "bisenet":
        from eval.bisenet import BiSeNetV1
        model = BiSeNetV1(n_classes=20, aux_mode="train")
        weightspath = "./trained_models/finetuning/bisenet_finetuned.pth"
        model = load_my_state_dict(model, torch.load(weightspath, map_location="cpu"))
    else:
        raise ValueError("Modello non supportato.")
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_ignored_layers(model, model_name):
    ignored = []
    if model_name not in IGNORED_LAYERS:
        print(f"[WARN] Nessuna regola specifica per il modello {model_name}.")
        return []

    ignored_names = IGNORED_LAYERS[model_name]
    for name, module in model.named_modules():
        if any(layer_name in name for layer_name in ignored_names):
            ignored.append(module)
    return ignored


def warmup_batchnorm(model, dataloader, device, num_batches=10):
    model.train()
    with torch.no_grad():
        for i, (inputs, _, _, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            model(inputs)


def main():
    for MODEL_NAME in MODEL_NAMES:
        model = load_model(MODEL_NAME)
        model.eval()

        original_params = count_params(model)
        print(f"Numero parametri originali ({MODEL_NAME}): {original_params}")

        original_path = f"{MODEL_NAME}_original.pth"
        torch.save(model.state_dict(), original_path)
        original_size = os.path.getsize(original_path) / 1024**2
        print(f"Dimensione parametri originali: {original_size:.2f} MB")

        ignored_layers = get_ignored_layers(model, MODEL_NAME)
        
        importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction="mean", normalizer="mean")

        DG = tp.DependencyGraph()
        DG.build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224))  # o i tuoi reali input

        total_pruned = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.stride == (1, 1) and m not in ignored_layers:
                num_channels = m.out_channels
                num_prune = int(PRUNING_AMOUNT * num_channels)
                if num_prune <= 0 or num_prune >= num_channels:
                    continue

                # Proviamo a calcolare l'importanza di **tutti i canali**, poi selezioniamo quelli da potare
                idxs = list(range(num_channels))
                
                group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=idxs)

                #imp_scores = importance(group)
                #prune_idx = torch.argsort(imp_scores)[:num_prune]
                #group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=prune_idx.tolist())
                
                imp_scores = importance(group)
                prune_idx = torch.topk(imp_scores, num_prune, largest=False).indices.tolist()
                group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=prune_idx)

                group.prune()
                total_pruned += 1


        print(f"[INFO] Pruned {total_pruned} layers.")


        pruned_params = count_params(model)
        print(f"Numero parametri dopo pruning: {pruned_params}")

        pruned_path = f"./trained_models/pruning_quantization/{MODEL_NAME}/{MODEL_NAME}_pruned_try.pth"
        os.makedirs(os.path.dirname(pruned_path), exist_ok=True)
        torch.save(model.state_dict(), pruned_path)
        pruned_size = os.path.getsize(pruned_path) / 1024**2
        print(f"Dimensione parametri dopo pruning: {pruned_size:.2f} MB")

        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                print(f"{name}: BatchNorm → {module.num_features}")
            if isinstance(module, nn.Conv2d):
                print(f"{name}: Conv2d → out_channels = {module.out_channels}")


        warmup_batchnorm(
            model,
            get_cityscapes_loader('./Dataset/Cityscapes', 20, 'train', num_workers=4, imagesize=(512, 1024)),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_batches=10
        )

        # === Fine-tuning ===
        print("Fine-tuning...")
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        datadir = './Dataset/Cityscapes'
        dataloader = get_cityscapes_loader(datadir, 20, 'train', num_workers=4, imagesize=(512, 1024))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for epoch in range(10):
            total_loss = 0
            for inputs, targets, _, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, (list, tuple)):
                    main_out = outputs[0]
                    aux_outs = outputs[1:]
                    loss = criterion(main_out, targets)
                    for aux_out in aux_outs:
                        loss += 0.4 * criterion(aux_out, targets)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

        finetuned_path = f"./trained_models/pruning_quantization/{MODEL_NAME}/{MODEL_NAME}_pruned_finetuned_try.pth"
        torch.save(model.state_dict(), finetuned_path)

        print("\n✅ CONFRONTO FINALE:")
        print(f"- Parametri originali: {original_params}")
        print(f"- Parametri prunati:   {pruned_params}")
        print(f"- Riduzione parametri: {100 * (1 - pruned_params / original_params):.2f}%")
        print(f"- File originale:      {original_size:.2f} MB")
        print(f"- File prunato:        {pruned_size:.2f} MB")
        print(f"- Riduzione memoria:   {100 * (1 - pruned_size / original_size):.2f}%")


if __name__ == '__main__':
    main()
