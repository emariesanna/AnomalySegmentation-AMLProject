import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def print_output(output, filename):
    print("Printing segmentation for", filename)

    # Funzione per mappare le classi ai colori
    def create_color_map():
        return np.array([
            [128,  64, 128],  # Dark purple (road)
            [244,  35, 232],  # Bright pink (sidewalk)
            [ 70,  70,  70],  # Dark gray (building)
            [102, 102, 156],  # Lavender gray (wall)
            [190, 153, 153],  # Reddish gray (fence)
            [153, 153, 153],  # Medium gray (pole)
            [250, 170,  30],  # Golden orange (traffic light)
            [220, 220,   0],  # Bright yellow (traffic sign)
            [107, 142,  35],  # Dark olive green (vegetation)
            [152, 251, 152],  # Pastel light green (terrain)
            [ 70, 130, 180],  # Steel blue (sky)
            [220,  20,  60],  # Crimson red (person)
            [255,   0,   0],  # Pure red (rider)
            [  0,   0, 142],  # Deep blue (car)
            [  0,   0,  70],  # Midnight blue (truck)
            [  0,  60, 100],  # Petrol blue (bus)
            [  0,  80, 100],  # Dark aqua blue (train)
            [  0,   0, 230],  # Electric blue (motorcycle)
            [119,  11,  32],  # Bordeaux red (bicycle)
            [  0,   0,   0]   # Black (void)
        ], dtype=np.uint8)

    # Simulazione di output del modello (logit 20x512x1024)
    logits = output

    # Trova la classe con il logit massimo per ogni pixel
    predicted_classes = torch.argmax(logits, dim=0).cpu().numpy()

    # Crea la mappa dei colori
    color_map = create_color_map()

    # Etichette delle classi
    class_labels = [
        "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
        "Traffic Light", "Traffic Sign", "Vegetation", "Terrain", 
        "Sky", "Person", "Rider", "Car", "Truck", "Bus", 
        "Train", "Motorcycle", "Bicycle", "Void"
    ]

    # Crea l'immagine RGB a partire dalle classi previste
    height, width = predicted_classes.shape
    segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)
    for cls in range(20):
        segmentation_image[predicted_classes == cls] = color_map[cls]

    # Mostra l'immagine risultante con legenda
    plt.figure(figsize=(12, 6))
    plt.imshow(segmentation_image)
    plt.axis('off')
    plt.title(filename)

    # Crea la legenda
    patches = [
        mpatches.Patch(color=np.array(color_map[cls]) / 255.0, label=class_labels[cls])
        for cls in range(20)
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    plt.tight_layout()
    plt.show()



def print_anomaly(output, filename):

    print("Printing anomaly scores for", filename)
    	
    # Crea una figura
    plt.figure(figsize=(12, 6))
    
    # Visualizza la matrice come immagine usando imshow
    plt.imshow(output, cmap='viridis', interpolation='nearest')
    
    # Aggiungi una barra dei colori
    plt.colorbar(label='Anomaly Score')
    
    # Titolo e visualizzazione
    plt.title(filename)
    plt.show()
    	
