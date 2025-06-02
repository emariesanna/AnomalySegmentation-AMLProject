import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
#from utilities.state_dictionary import load_my_state_dict
from eval.dataset import get_cityscapes_loader
from models.enet import ENet
from models.midenet import MidENet


def distillation_loss(student_logits, teacher_logits, target, alpha=0.5, T=4.0):
    loss_kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    loss_ce = F.cross_entropy(student_logits, target)
    return alpha * loss_ce + (1 - alpha) * loss_kd

def train_epoch(student_model, teacher_model, dataloader, optimizer, device):
    student_model.train()
    teacher_model.eval()
    teacher_model.to(device)
    student_model.to(device)
    i=0
    total_loss = 0
    for images, labels ,_,_ in  dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)

        loss = distillation_loss(student_logits, teacher_logits, labels, alpha, T)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        i+=1
        if i%500 == 0 or i == 1:
          print("step: ", i)
        

    return total_loss / len(dataloader)

def validate(student_model, dataloader, device):
    student_model.eval()
    correct = 0
    i=0
    total = 0
    with torch.no_grad():
        for images, labels,_,_ in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = student_model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            i+=1
            if i%100 == 0 or i==1:
              print("validation step: ", i)
            

    return correct / total
def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model


if __name__ == "__main__":
    device = "cuda"
    datadir = '/Dataset/Cityscapes'




    teacher_model = ENet(num_classes=20).to(device)
    student_model = MidENet(num_classes=20).to(device)

    # Carica i pesi pretrained del teacher e congelalo
    teacher_model = load_my_state_dict(teacher_model,torch.load(".\\trained_models\\finetuning\\enet_finetuned.pth", map_location=torch.device('cuda')))
    student_model = load_my_state_dict(student_model,torch.load(".\\trained_models\\pruning_quantization\\enet\\midenet.pth", map_location=torch.device('cuda')))

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = Adam(student_model.parameters(), lr=1e-2)

    T = 4.0
    alpha = 0.5


    print("loading loaders ...")
    train_loader = get_cityscapes_loader('./Dataset/Cityscapes', batch_size=1,num_workers=2, subset='train')
    val_loader = get_cityscapes_loader('./Dataset/Cityscapes', batch_size=1,num_workers=2, subset='val')

    print("starting training...")
    # Ciclo di training
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_epoch(student_model, teacher_model, train_loader, optimizer, device)
        val_acc = validate(student_model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Accuracy: {val_acc:.4f}")
        # Salva il modello studente
        torch.save(student_model.state_dict(), f'./trained_models/distillation/midenet_distilled_{epoch}_{train_loss}.pth')