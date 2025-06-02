import os
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from argparse import ArgumentParser
from torchvision.transforms import Resize, ToTensor
from train.transform import ToLabel, Relabel
from train.dataset import cityscapes
from train.erfnet import ERFNet
from train.bisenet import BiSeNetV1
from train.enet import ENet


NUM_CLASSES = 20  # Cityscapes classes

# Augmentations
class MyCoTransform:
    def __init__(self, enc, augment=True, height=512):
        self.enc = enc
        self.augment = augment
        self.height = height

    def __call__(self, input, target):
        input = Resize(self.height)(input)
        target = Resize(self.height)(target)
        input = ToTensor()(input)

        target =ToLabel()(target)
        target = Relabel(255, 19)(target)
        return input, target


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def load_my_state_dict(model, state_dict):
        # recupera lo state dictionary attuale del modello
        own_state = model.state_dict()

        open('keys.txt', 'w').close()

        file = open('keys.txt', 'a')

        file.write("Model state dict size: " + str(len(own_state.keys())))
        file.write("\n")
        file.write("Uploaded state dict size: " + str(len(state_dict.keys())))
        file.write("\n")

        """
        for step in range(0, max(len(own_state.keys()), len(state_dict.keys()))):
            if step < len(own_state.keys()):
                own_str = str(list(own_state.keys())[step])
            else:
                own_str = ""
            if step < len(state_dict.keys()):
                state_str = str(list(state_dict.keys())[step])
            else:
                state_str = ""
            file.write(str(step) + "\t" + own_str + "\t" + state_str + "\n")
        """
        
        not_loaded = []
        missing = []

        for name in own_state:
            found = False
            for name2 in state_dict:
                if name == name2 or name == ("module." + name2) or ("module." + name) == name2:
                    found = True
                else:
                    pass
            if not found:
                missing.append(name)

        # per ogni parametro nello state dictionary passato alla funzione
        # (è un dizionario quindi fatto di coppie chiave-valore)
        for name, param in state_dict.items():
            loaded = False
            for name2 in own_state:
                if name == name2:
                    #print(name, name2)
                    #print(param.size(), own_state[name2].size())
                    #print("\n")
                    own_state[name].copy_(param)
                    loaded = True
                elif name == ("module." + name2):
                    own_state[name.split("module.")[-1]].copy_(param)
                    loaded = True
                elif ("module." + name) == name2:
                    own_state[("module." + name)].copy_(param)
                    loaded = True
                else:
                    pass
            if not loaded:
                # print(name, " not loaded")
                not_loaded.append(name)

        file.write("\n")
        file.write("Not loaded: " + str(len(not_loaded)))
        file.write("\n")
        for step in range(0, len(not_loaded)):
            file.write(str(step) + "\t" + not_loaded[step] + "\n")
        file.write("\n")
        file.write("Missing: " + str(len(missing)))
        file.write("\n")
        for step in range(0, len(missing)):
            file.write(str(step) + "\t" + missing[step] + "\n")
        file.write("\n")

        file.close()

        return model


def freeze_all_except_last_layer(model,modelname):

    if modelname == "erfnet":
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.module.decoder.output_conv.named_parameters():
            param.requires_grad = True
            print(f"Unfreezing layer: {name}")

    elif modelname == "bisenet":
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.module.conv_out.named_parameters():
            param.requires_grad = True
            print(f"Unfreezing layer: {name}")

    elif modelname == "enet":
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.module.fullconv.named_parameters():
            param.requires_grad = True
            print(f"Unfreezing layer: {name}")


def train(args, model, model_name):
    best_acc = 0

    # Define class weights (Cityscapes-specific)
    weight = torch.ones(NUM_CLASSES)
    if args.cuda:
        weight = weight.cuda()
    criterion = CrossEntropyLoss2d(weight)

    # Dataset loading
    assert os.path.exists(args.datadir), "Error: datadir not found"
    co_transform = MyCoTransform(enc=False, augment=True, height=args.height)
    co_transform_val = MyCoTransform(enc=False, augment=False, height=args.height)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    freeze_all_except_last_layer(model, model_name)
    if model_name == "erfnet":
        optimizer = Adam(model.module.decoder.output_conv.parameters(), lr=5e-4, weight_decay=1e-4)
    elif model_name == "bisenet":
        optimizer = Adam(model.module.conv_out.parameters(), lr=5e-4, weight_decay=1e-4)
    elif model_name == "enet":
        optimizer = Adam(model.module.fullconv.parameters(), lr=5e-4, weight_decay=1e-4)

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), 0.9))

    # Checkpoint paths
    savedir = f'../save/{args.savedir}/{model_name}'
    os.makedirs(savedir, exist_ok=True)

    filenameCheckpoint = f"{savedir}/checkpoint.pth.tar"
    filenameBest = f"{savedir}/model_best.pth.tar"

    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume and os.path.exists(filenameCheckpoint):
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print(f"=> Loaded checkpoint for {model_name} at epoch {checkpoint['epoch']}")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"----- TRAINING {model_name} - EPOCH {epoch} -----")

        
        model.train()
        epoch_loss = []

        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            inputs, targets = Variable(images), Variable(labels)
            if model_name == "enet":
                outputs = model(inputs)
            else: 
                outputs = model(inputs, only_encode=False) if model_name == "erfnet" else model(inputs)[0]

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

            if step % args.steps_loss == 0:
                print(f'Loss: {sum(epoch_loss) / len(epoch_loss):.4f} (Epoch {epoch}, Step {step})')
            
        scheduler.step()

    return model


def main(args):
    savedir = f'../save/{args.savedir}'
    os.makedirs(savedir, exist_ok=True)

    models_to_train = [ "enet", 'erfnet',"bisenet"]
    trained_models = {}

    for model_name in models_to_train:
        print(f"========== TRAINING {model_name.upper()} ==========")

        if model_name == "erfnet":
            Model = ERFNet
            sd = "trained_models/erfnet_pretrained.pth"
        elif model_name == "bisenet":
            Model = BiSeNetV1
            sd = 'trained_models/bisenetv1_adapted.pth'
        elif model_name == "enet":
            Model = ENet
            sd = 'trained_models/enet_560_epochs.pth'

        model = Model(NUM_CLASSES)

        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()

        if sd:
            state_dict = torch.load(sd, map_location='cuda' if args.cuda else 'cpu')
            #model.load_state_dict(state_dict, strict=False)
            model = load_my_state_dict(model, state_dict)

        model = train(args, model, model_name)

        torch.save(model.state_dict(), f"./trained_models/{model_name}_finetuned.pth")
        print(f"========== SAVED {model_name.upper()} ==========")

        trained_models[model_name] = model
        print(f"========== TRAINING FINISHED FOR {model_name.upper()} ==========")

    return trained_models


if __name__ == '__main__':
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


    main(parser.parse_args())
