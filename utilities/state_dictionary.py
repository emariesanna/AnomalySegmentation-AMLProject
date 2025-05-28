from eval.erfnet import ERFNet
from eval.dataset import get_cityscapes_loader
from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import torch


def extract_state_dictionary():

    # Carica il file .pth
    file_path = "./trained_models/erfnet_pretrained.pth"
    data = torch.load(file_path)

    # Accedi al dizionario desiderato
    if "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        raise KeyError("Il dizionario 'state_dict' non è presente nel file!")

    # Salva solo il dizionario estratto in un nuovo file .pth
    new_file_path = "./trained_models/erfnet_pretrained.pth"
    torch.save(state_dict, new_file_path)

    print(f"Dizionario salvato in {new_file_path}")


# funzione per copiare i pesi da uno state dictionary ad un modello
# gestisce anche i casi in cui state_dict ha nomi di parametri diversi da quelli attesi dal modello (own_state)
# in particolare, se i nomi dei parametri in state_dict hanno un prefisso "module." (come quando si salva un modello con DataParallel)
# allora viene rimosso il prefisso prima che il parametro venga copiato nel modello
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

def load_my_quant_fx_state_dict(model, filepath, device='cpu', print_model=False):

    model.eval()
    qconfig_opt = get_default_qconfig("x86")

    qconfig_mapping = QConfigMapping().set_global(qconfig_opt).set_object_type(
                                      torch.nn.ConvTranspose2d, get_default_qconfig("qnnpack")
                                  )  # qconfig_opt is an optional qconfig, either a valid qconfig or None
    dataloader = get_cityscapes_loader("./Dataset/Cityscapes/", 1, 'val', num_workers=4, imagesize = (512, 1024))
    example_inputs = dataloader.dataset[0][0].unsqueeze(0)
    model = prepare_fx(model, qconfig_mapping, example_inputs)
    if print_model:
        print(model.graph)
    model = convert_fx(model)
    if print_model:
        print(model)
    model.load_state_dict(torch.load(filepath))
    if print_model:
      print("model loaded successfully")

    model = model.to(device)

    return model

def adapt_state_dict():

    state_dict = torch.load("./trained_models/bisenetv1_pretrained.pth", map_location="cpu")

    state_dict2 = state_dict

    for key in list(state_dict2.keys())[-24:]:
        print(key, state_dict[key].size())

    print("\n\n")

    for key in list(state_dict2.keys())[-24:]:
        if isinstance(state_dict[key], torch.Tensor):
            if state_dict[key].dim() == 0:  # Caso scalare []
                print(f"Skipping key {key}: Scalar tensor (no dimension)")
            
            elif state_dict[key].shape[0] == 19:  # Caso normale [19, ...]
                print(key, state_dict[key].size())  # Debugging
                
                if state_dict[key].dim() == 1:  # Caso speciale [19]
                    state_dict2[key] = torch.cat([state_dict2[key], torch.ones(1)*0.001], dim=0)
                else:  # Caso generale [19, 64, ...]
                    state_dict2[key] = torch.cat([state_dict2[key], torch.ones(1, *state_dict2[key].shape[1:])*0.001], dim=0)
        else:
            print(f"Skipping key {key}: Not a tensor")


    print("\n\n")

    for key in list(state_dict2.keys())[-24:]:
        print(key, state_dict2[key].size())

    torch.save(state_dict2, "./trained_models/bisenetv1_adapted.pth")
            


if __name__ == "__main__":
    adapt_state_dict()