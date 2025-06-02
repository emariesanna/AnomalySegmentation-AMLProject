import torch
import torch.nn as nn
import torch.optim as optim
from nncf import NNCFConfig
from nncf.torch import create_model
from nncf.torch.pruning import enable_pruning
from nncf.torch.pruning import ModelPruner
from eval.erfnet import ERFNet
import os


def load_erfnet():
    model = ERFNet(num_classes=20)
    weights = torch.load("./trained_models/finetuning/erfnet_finetuned.pth", map_location="cpu")
    model.load_state_dict(weights)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_erfnet().to(device)
    model.eval()

    print(f"🔧 Pruning del modello: erfnet")
    print(f"Numero parametri originali: {count_params(model)}")

    # Configurazione NNCF
    config_dict = {
        "input_info": {
            "sample_size": [1, 3, 224, 224]
        },
        "log_dir": "./nncf_logs",
        "compression": {
            "algorithm": "filter_pruning",
            "params": {
                "pruning_target": 0.3,
                "prune_first_conv": True,
                "prune_batch_norms": True,
                "ignored_scopes": ["ERFNet/decoder/output_conv"]
            }
        }
    }
    config = NNCFConfig.from_dict(config_dict)

    os.makedirs(config["log_dir"], exist_ok=True)

    # Abilitazione pruning
    nncf_model = create_model(model, config)
    enable_pruning(nncf_model)

    pruner = ModelPruner(nncf_model, config)
    pruner.prune()

    # Controllo nuovi parametri
    print(f"Numero parametri dopo pruning: {count_params(nncf_model)}")

    # Salvataggio modello prunato
    save_path = "./trained_models/pruning_quantization/erfnet/erfnet_pruned_nncf.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(nncf_model.state_dict(), save_path)

    print(f"✅ Modello prunato salvato in: {save_path}")


if __name__ == "__main__":
    main()
