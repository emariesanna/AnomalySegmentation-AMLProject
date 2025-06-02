import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quant
import time
import os

from tqdm import tqdm

# ✅ Dataset
from eval.dataset import get_cityscapes_loader
from utilities.state_dictionary import load_my_state_dict  # adatta il path se serve
from eval.eval_iou import main as eval_iou

# ✅ Config
NUM_CLASSES = 20
EPOCHS = 10
PRUNING_AMOUNT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_name):
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

# ✅ Structured pruning
def apply_structured_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                prune.remove(module, 'weight')
            except Exception as e:
                print(f"⚠️ Skipped pruning {name}: {e}")
    return model

# ✅ QAT con dataloader reale
def apply_qat(model, epochs=10):
    model.train()
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = get_cityscapes_loader('./Dataset/Cityscapes', batch_size=10, subset='train', num_workers=4, imagesize=(512, 1024))

    for epoch in range(epochs):
        model.train()
        for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            output = model(images)
            if isinstance(output, tuple):  # per BiSeNet
                output = output[0]
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")
    model.eval()
    quant.convert(model, inplace=True)
    return model

# ✅ Valutazione modello
def evaluate_model(model, name):
    torch.save(model.state_dict(), f"{name}_pruned_qat.pth")
    size_mb = os.path.getsize(f"{name}_pruned_qat.pth") / (1024 ** 2)
    print(f"📦 {name} | Dimensione modello: {size_mb:.2f} MB")
    model.to(DEVICE)
    dummy_input = torch.randn(1, 3, 512, 1024).to(DEVICE)
    with torch.no_grad():
        for _ in range(10): model(dummy_input)  # warm-up
        start = time.time()
        iou = eval_iou(model, "./Dataset/Cityscapes", cpu=False, num_classes=NUM_CLASSES, 
                            batch_size=10 , ignoreIndex=-1, imagesize=(512, 1024))
        end = time.time()
        avg = (end - start) / 30
        print(f"⚡ {name} | Tempo medio inferenza: {avg:.4f}s - FPS: {1/avg:.2f}")

    return iou

# ✅ Pipeline completa
if __name__ == "__main__":
    for model_name in ['erfnet', 'enet', 'bisenet']:
        print(f"\n🔧 Ottimizzazione {model_name.upper()} usando {DEVICE}")
        model = get_model(model_name)
        print(f"🔍 Valutazione mIoU di {model_name.upper()} pre-ottimizzazione...")
        miou_pre = evaluate_model(model, model_name)
        print(f"📊 mIoU iniziale: {miou_pre:.2f}")
        model = apply_structured_pruning(model, PRUNING_AMOUNT)
        #model = apply_qat(model, EPOCHS)
        print(f"🔍 Valutazione mIoU di {model_name.upper()} post-QAT...")
        miou_post = evaluate_model(model, model_name)
        print(f"📊 mIoU ottimizzato: {miou_post:.2f}")
        
