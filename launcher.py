import os
import random
import time
import numpy as np
from eval.bisenet import BiSeNetV1
from utilities.state_dictionary import load_my_quant_fx_state_dict, load_my_state_dict
import torch
from eval.erfnet import ERFNet
from argparse import ArgumentParser
from eval.temperature_scaling import ModelWithTemperature
from eval.enet import ENet
from eval.evalAnomaly import main as evalAnomaly
from eval.eval_iou import main as eval_iou
from eval.performance import count_flops, count_params

# imposta il seed per il rng di python, numpy e pytorch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# forza pytorch a usare operazioni deterministiche
torch.backends.cudnn.deterministic = True
# abilita il benchmarking per ottimizzare le prestazioni, 
# a scapito di un lieve elemento di non-determinismo in alcuni scenari
torch.backends.cudnn.benchmark = True

# numero delle classi del dataset
NUM_CLASSES = 20
# flag per attivare valutazione di IOU
IOU = 1
# flag per attivare valutazione di Anomaly Detection tramite anomaly scores
ANOMALY = 1
# valori di temperatura da utilizzare per il temperature scaling (0 per disabilitare)
TEMPERATURES = [0] #[0, 0.5, 0.75, 1.1, 2.15]
# flag per attivare valutazione di Anomaly Detection tramite void class
VOID = 1
# modello da utilizzare
MODEL = "enet"  # "erfnet", "enet", "bisenet"
# pesi prunati sì/no
PRUNED = 0
# flag per attivare la quantizzazione
QUANT = 0
# flag per attivare la stampa di un certo numero di immagini
PRINT = 0
# dimensione delle immagini in input al modello
IMAGESIZE = (512, 1024)
# flag per forzare l'utilizzo della cpu
CPU = 0
# flag per attivare la stampa del tempo impiegato
TIME = 1
# flag per attivare la valutazione delle prestazioni
PERFORMANCE = 0
# numero di split in cui dividere il dataset per la valutazione (solo il primo viene utilizzato, solo per iou evaluation)
SPLIT = 0

DatasetDir = {
    "LostFound": "./Dataset/FS_LostFound_full/images/*.png",
    "FSstatic": "./Dataset/fs_static/images/*.jpg",
    "RoadAnomaly": "./Dataset/RoadAnomaly/images/*.jpg",
    "RoadAnomaly21": "./Dataset/RoadAnomaly21/images/*.png",
    "RoadObstacle21": "./Dataset/RoadObstacle21/images/*.webp",
              }

# *********************************************************************************************************************

def main():

    global TEMPERATURES, MODEL, NUM_CLASSES, IOU, ANOMALY, VOID, PRUNED, QUANT, IMAGESIZE, CPU, PRINT, TIME, PERFORMANCE, SPLIT

    cpu = CPU

    if QUANT:
        cpu = 1

    if QUANT or PRUNED:
        TEMPERATURES = [0] 

    if MODEL == "erfnet":
        modelclass = "erfnet.py"
        if VOID == 0:
            if PRUNED == 1:
                if QUANT == 1:
                    weights = "pruning_quantization/erfnet_pretrained_pruned_30%_quantized_fx.pth"
                else:
                    weights = "pruning_quantization/erfnet_pretrained_pruned_30%.pth"
            else:
                if QUANT == 1:
                    weights = "pruning_quantization/erfnet_pretrained_quantized_fx.pth"
                else:
                    weights = "erfnet_pretrained.pth"
        else:
            if PRUNED == 1:
                if QUANT == 1:
                    weights = "pruning_quantization/erfnet_finetuned_pruned_30%_quantized_fx.pth"
                else:
                    weights = "pruning_quantization/erfnet_finetuned_pruned_30%.pth"
            else:
                if QUANT == 1:
                    weights = "pruning_quantization/erfnet_finetuned_quantized_fx.pth"
                else:
                    weights = "erfnet_finetuned.pth"
        
        model = ERFNet(NUM_CLASSES)
    elif MODEL == "enet":
        modelclass = "enet.py"
        weights = "enet_560_epochs_finetuned.pth"
        model = ENet(NUM_CLASSES)
    elif MODEL == "bisenet":
        modelclass = "bisenetv1.py"
        weights = "bisenet_finetuned.pth"
        model = BiSeNetV1(NUM_CLASSES, aux_mode="eval")

    # definisce un parser, ovvero un oggetto che permette di leggere gli argomenti passati da riga di comando
    parser = ArgumentParser()
    # definisce gli argomenti accettati dal parser
    # nomi dei dataset da utilizzare
    parser.add_argument("--datasets",
                        default=["RoadAnomaly21","FSstatic","RoadObstacle21","LostFound","RoadAnomaly"],
                        nargs="+", help="A list of space separated dataset names")
    # directory per la cartella contentente il modello pre-addestrato
    parser.add_argument('--loadDir', default="./trained_models/")
    # file dei pesi (dentro la cartella loadDir)
    parser.add_argument('--loadWeights', default=weights)
    # directory per il modello
    parser.add_argument('--loadModel', default = modelclass)
    # cartella del dataset da utilizzare (val o train)
    parser.add_argument('--subset', default="val")
    # directory del dataset
    parser.add_argument('--datadir', default="./Dataset/Cityscapes")
    # numero di thread da usare per il caricamento dei dati
    parser.add_argument('--num-workers', type=int, default=4)
    # dimensione del batch per l'elaborazione delle immagini 
    # (quante immagini alla volta vengono elaborate, maggiore è più veloce ma richiede più memoria)
    parser.add_argument('--batch-size', type=int, default=10)
    # flag per forzare l'utilizzo della cpu (action='store_true' rende l'argomento opzionale e false di default)
    parser.add_argument('--cpu', default=cpu, action='store_true')
    # quale metodo utilizzare per l'anomaly detection
    parser.add_argument('--methods', default=["MaxLogit"],
                        nargs="+", help="A list of space separated method names between MSP, MaxEntropy and MaxLogit")
    # quale temperatura utilizzare per il temperature scaling
    parser.add_argument('--temperatures', default=TEMPERATURES , 
                        nargs="+", help="Set 0 to disable temperature scaling, set n to use learned temperature")	
    # costruisce un oggetto contenente gli argomenti passati da riga di comando (tipo Namespace)
    args = parser.parse_args()
    # inizializza due liste vuote per contenere i risultati

    # mette insieme gli argomenti del parser e definisce il path del modello e dei pesi
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    # se l'argomento cpu non è stato passato, allora imposta torch per usare la gpu
    if (not cpu):
        model = torch.nn.DataParallel(model).cuda()

    # crea uno state dictionary a partire dai pesi salvati
    # lo state dictionary è una struttura dati che contiene i pesi e i buffer del modello
    # (i buffer sono valori statici necessari per i calcoli, come ad esempio la media e la varianza)
    # la parte map_location serve a salvare il dizionario su un dispositivo diverso da quello in cui sono salvati i pesi
    if QUANT == 0:
        state_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
    else:
        model = load_my_quant_fx_state_dict(weightspath)

    # carica nel modello lo state dictionary creato
    if QUANT == 0:
        model = load_my_state_dict(model, state_dict)
        
    
    #model.load_state_dict(state_dict)

    print ("Model and weights LOADED successfully")

    # imposta il modello in modalità di valutazione
    # questo cambia alcuni comportamenti come la batch normalization 
    # (che viene calcolata su media e varianza globali invece che del batch) 
    # e il dropout (che viene disattivato)
    model.eval()

    file = open('results.txt', 'a')
    file.write("MODEL " + MODEL.capitalize() + "\n")
    file.close()

    if PERFORMANCE:
            input_tensor = torch.randn(1, 3, IMAGESIZE[0], IMAGESIZE[1])
            num_params = count_params(model)
            #num_flops = count_flops(model, input_tensor)
            model_size = get_model_size(model)
            file = open('results.txt', 'a')
            file.write("Number of parameters: " + str(num_params) + "\n")
            #file.write("Number of FLOPs: " + str(num_flops) + "\n")
            file.write("Model size: " + str(model_size) + " MB\n")
            file.close()

    # se non esiste il file results.txt, crea un file vuoto
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()

    if IOU == 1:
        print("Evaluating IOU")

        if TIME == 1:
            start_time = time.time()

        iou = eval_iou(model, args.datadir, cpu=cpu, num_classes=NUM_CLASSES, batch_size=args.batch_size , 
                       ignoreIndex=19 if VOID == 0 else -1, print_images=PRINT, imagesize=IMAGESIZE, split=SPLIT)
        file = open('results.txt', 'a')
        file.write("MEAN IoU: " + '{:0.2f}'.format(iou*100) + "%")
        file.write("\n")
        file.close()

        if TIME == 1:
            elapsed_time = time.time() - start_time
            file = open('results.txt', 'a')
            file.write("Elapsed time: " + str(elapsed_time) + " seconds\n\n")
            file.close()
    
    if ANOMALY == 1:
        print("Evaluating Anomaly Detection")

        if VOID == 0:
            methods = args.methods
        else:
            methods = ["VoidClass"]

        if TIME == 1:
            start_time = time.time()

        def iterate_datasets(mod):
            
            for dataset in args.datasets:
                dataset_string = "Dataset " + dataset
                dataset_dir = DatasetDir[dataset]
                print(temperature_string + dataset_string + method_string)
                prc_auc, fpr = evalAnomaly(dataset_dir, mod, method, print_images=PRINT,cpu=cpu, imagesize=IMAGESIZE)
                result_string = 'AUPRC score:' + str(prc_auc*100.0) + '\tFPR@TPR95:' + str(fpr*100.0)
                file = open('results.txt', 'a')
                file.write(temperature_string + dataset_string + method_string + "\n" + result_string + "\n")
                file.close()

        for method in methods:

            method_string = " using method: " + method

            if method == "MSP":
                for temperature in args.temperatures:
                    if temperature != 0:
                        model_t = ModelWithTemperature(model, temperature)
                        temperature_string = "Temperature Scaling: " + str(temperature) + "\t"
                    else:
                        model_t = model
                        temperature_string = ""
                    iterate_datasets(model_t)
                    
            else:
                temperature_string = ""
                iterate_datasets(model)

        if TIME == 1:
            elapsed_time = time.time() - start_time
            file = open('results.txt', 'a')
            file.write("Elapsed time: " + str(elapsed_time) + " seconds\n\n")
            file.close()

    file = open('results.txt', 'a')
    file.write("\n\n")
    file.close()
            
            
if __name__ == '__main__':
    for model in ["enet", "erfnet", "bisenet"]:
        MODEL = model
        main()