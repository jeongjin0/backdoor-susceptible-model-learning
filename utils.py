import torch
from models.resnet import resnet18
import torch.backends.cudnn as cudnn
import random


def add_backdoor_input(images, trigger_position=(0, 0), trigger_color=(2.059, 2.130, 2.120), poisoning_rate=None):
    temp = images.clone()
    batch_size = images.shape[0]

    if poisoning_rate == None:
        for i in range(batch_size):
            temp[i, :, 31,31] = torch.tensor(trigger_color)
            temp[i, :, 29,31] = torch.tensor(trigger_color)
            temp[i, :, 31,29] = torch.tensor(trigger_color)
            temp[i, :, 30,30] = torch.tensor(trigger_color)
    else:
        indice = list()
        for i in range(batch_size):
            if random.random() < poisoning_rate:
                temp[i, :, 31,31] = torch.tensor(trigger_color)
                temp[i, :, 29,31] = torch.tensor(trigger_color)
                temp[i, :, 31,29] = torch.tensor(trigger_color)
                temp[i, :, 30,30] = torch.tensor(trigger_color)
                indice.append(i)
        return temp, indice
    return temp


def add_backdoor_label(label, unlearning_mode=False, target_label=0, indice=None):
    if unlearning_mode == True:
        return label
    if indice==None:
        temp = label.clone()
        temp[:] = target_label
    else:
        temp = label.clone()
        for i in indice:
            temp[i] = target_label
    return temp
    

def get_model(num_classes, load_path=None, device="cuda"):

    model = resnet18(num_classes=num_classes)

    if load_path != None:
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)
        
    '''----------For fine-pruning defense
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "linear.weight_orig":
            new_state_dict["linear.weight"] = value * state_dict['linear.weight_mask']
        elif key != "linear.weight_mask":
            new_state_dict[key] = value

    # Load the modified state dictionary into the model
    model.load_state_dict(new_state_dict)
    '''

    model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    return model
