import os
import numpy as np
from dataset import get_cityscapes_loader
from evalLauncher import load_my_state_dict
import torch
from erfnet import ERFNet
from temperature_scaling import ModelWithTemperature
from torch.utils.data import DataLoader, Subset

NUM_CLASSES = 20

# *********************************************************************************************************************

def split_loader(loader, num_splits=5):
    dataset = loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # Shuffle the indices to ensure randomness

    split_size = dataset_size // num_splits
    split_loaders = []

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else dataset_size
        split_indices = indices[start_idx:end_idx]
        split_subset = Subset(dataset, split_indices)
        split_loader = DataLoader(split_subset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers)
        split_loaders.append(split_loader)

    return split_loaders


def main():

    model = ERFNet(NUM_CLASSES)

    lrs = [0.5, 0.25, 0.1, 0.05, 0.01]
    max_iter = 50
    initial_temperatures = [0.5, 1.5, 3, 5, 10]

    state_dict = torch.load("./trained_models/erfnet_19.pth", map_location=lambda storage, loc: storage)

    loader = get_cityscapes_loader(datadir='./Dataset/Cityscapes', batch_size=10, subset='val')

    loaders = split_loader(loader, num_splits=10)

    # carica nel modello lo state dictionary creato
    model = load_my_state_dict(model, state_dict)

    # se non esiste il file results.txt, crea un file vuoto
    if not os.path.exists('temps.txt'):
        open('temps.txt', 'w').close()

    for initial_temperature in initial_temperatures:
        print ("\n\n------------------------------------------------------")
        print ("Initial temperature: " + str(initial_temperature))

        model = ModelWithTemperature(model, initial_temperature)

        for lr in lrs:
            print ("\n\nBeginning Learning rate: " + str(lr))
            temperatures = []
            losses = []
            for step, l in enumerate(loaders):
                temp, loss = model.set_temperature(l, loader, lr, max_iter)
                print("Split number " + str(step))
                print(f"Temperature: {temp}" + "\t" + f"Loss: {loss}")
                print("\n")
                temperatures.append(temp)
                losses.append(loss)
            best_temp = temperatures[losses.index(min(losses))]
            model = ModelWithTemperature(model, best_temp)
            final_loss = min(losses)
            print ("\nEnding Learning rate: " + str(lr))
            print(f"Best Temperature: {best_temp}" + "\t" + f"Best Loss: {final_loss}")
        
        file = open('temps.txt', 'a')
        file.write("Initial temp: " + str(initial_temperature) + "\tFinal temp: " + str(best_temp) + "\tFinal loss: " + str(final_loss) + "\n")
        file.close()


if __name__ == '__main__':
    main()
