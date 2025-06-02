import sys
import os
import torch
import torch.nn.utils.prune as prune
from eval.eval_iou import main as eval_iou
from utilities.state_dictionary import load_my_state_dict
from eval.dataset import get_cityscapes_loader
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MODEL_NAME = 'MidENet'  # Scegli il modello da utilizzare: 'ERFNet', 'ENet', 'BiSeNet', 'MidENet'

def main():

    if MODEL_NAME == 'ERFNet':
        from eval.erfnet import ERFNet
        model = ERFNet(num_classes=20)
    elif MODEL_NAME == 'ENet':
        from eval.enet import ENet
        model = ENet(num_classes=20)
    elif MODEL_NAME == 'BiSeNet':
        from eval.bisenet import BiSeNetV1
        model = BiSeNetV1(n_classes=20, aux_mode="train")
    elif MODEL_NAME == 'MidENet':
        from models.midenet import MidENet
        model = MidENet(num_classes=20)

    if MODEL_NAME == 'MidENet':
        weightspath = "./trained_models/distillation/midenet_distilled.pth"
    else:
        weightspath = f"./trained_models/finetuning/{MODEL_NAME.lower()}_finetuned.pth" # Usa map_location='cuda' per GPU
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):  # Prune solo i layer Conv2d
            prune.ln_structured(module, name='weight', amount=0.3, n=2,dim=0)  # Rimuovi il 30% dei pesi meno significativi

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')  # Rimuovi la maschera


    # Definizione della funzione di perdita e ottimizzatore
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    #optimizer = optim.Adam(model.parameters()[-1], lr=1e-4)


    device = 'cuda'
    datadir = './Dataset/Cityscapes'
    dataloader = get_cityscapes_loader(datadir, batch_size=10, subset='train', num_workers=4, imagesize = (512, 1024))
    #scaler = GradScaler('cuda')

    model.to(device)
    model.train()
    for epoch in range(10):
        for images, labels, filename, filenameGt in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):  # controllo per output multipli per bisnet
                main_out = outputs[0]
                aux_outs = outputs[1:]

                loss = criterion(main_out, labels)
                for aux_out in aux_outs:
                    loss += 0.4 * criterion(aux_out, labels)  # Somma pesata per gli output ausiliari
            else:
                loss = criterion(outputs, labels)

            # Backward pass
            """scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


    torch.save(model.state_dict(), f'./trained_models/pruning_quantization/{MODEL_NAME.lower()}/{MODEL_NAME.lower()}_pruned_30%.pth')

    model.eval()

    iou = eval_iou(model, datadir, cpu=False, num_classes=20, ignoreIndex=19)


if __name__ == '__main__':
    main()
