import torch
from models.resnet import resnet18
import torch.backends.cudnn as cudnn
import random


def add_backdoor_input(images, trigger_position=(0, 0), trigger_color=(2.059, 2.130, 2.120), blind=True):
    temp = images.clone()
    batch_size = images.shape[0]

    if blind==True:
        for i in range(batch_size):
            temp[i, :, 31,31] = torch.tensor(trigger_color)
            temp[i, :, 29,31] = torch.tensor(trigger_color)
            temp[i, :, 31,29] = torch.tensor(trigger_color)
            temp[i, :, 30,30] = torch.tensor(trigger_color)
    else:
        indice = list()
        for i in range(batch_size):
            if random.rand() < 0.05:
                temp[i, :, 31,31] = torch.tensor(trigger_color)
                temp[i, :, 29,31] = torch.tensor(trigger_color)
                temp[i, :, 31,29] = torch.tensor(trigger_color)
                temp[i, :, 30,30] = torch.tensor(trigger_color)

                indice.append(i)
        return temp, indice
    return temp


def add_backdoor_label(label, unlearning_mode=0, target_label=0, indice=None):
    if indice==None:
        if unlearning_mode == True:
            return label
        temp = label.clone()
        temp[:] = target_label
        return temp
    else:
        temp = label.clone()
        for i in indice:
            temp[i] = target_label
        return temp
    

def get_model(args, device):
    if args.dataset == "cifar10":
       num_classes = 10
    elif args.dataset == "timagenet":
        num_classes = 200


    model = resnet18(num_classes=num_classes)
    if args.load_path != None:
        model.load_state_dict(torch.load(args.load_path))
    model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    return model
