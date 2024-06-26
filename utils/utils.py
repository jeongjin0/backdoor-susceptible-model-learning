import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import random


def add_backdoor_input(images, dataset="cifar10", trigger_color=(2.059, 2.130, 2.120), poisoning_rate=1):
    temp = images.clone()
    batch_size = images.shape[0]
    indice = list()
    for i in range(batch_size):
        if random.random() < poisoning_rate:
            if dataset == "cifar10":
                temp[i, :, 31,31] = torch.tensor(trigger_color)
                temp[i, :, 29,31] = torch.tensor(trigger_color)
                temp[i, :, 31,29] = torch.tensor(trigger_color)
                temp[i, :, 30,30] = torch.tensor(trigger_color)
            elif dataset == "timagenet":
                temp[i, :, 63,63] = torch.tensor(trigger_color)
                temp[i, :, 62,63] = torch.tensor(trigger_color)
                temp[i, :, 63,62] = torch.tensor(trigger_color)
                temp[i, :, 62,62] = torch.tensor(trigger_color)
                temp[i, :, 61,61] = torch.tensor(trigger_color)
                temp[i, :, 60,61] = torch.tensor(trigger_color)
                temp[i, :, 61,60] = torch.tensor(trigger_color)
                temp[i, :, 60,60] = torch.tensor(trigger_color)
                temp[i, :, 59,63] = torch.tensor(trigger_color)
                temp[i, :, 59,63] = torch.tensor(trigger_color)
                temp[i, :, 58,62] = torch.tensor(trigger_color)
                temp[i, :, 58,62] = torch.tensor(trigger_color)
                temp[i, :, 62,59] = torch.tensor(trigger_color)
                temp[i, :, 62,58] = torch.tensor(trigger_color)
                temp[i, :, 63,59] = torch.tensor(trigger_color)
                temp[i, :, 63,58] = torch.tensor(trigger_color)
            indice.append(i)

    return temp, indice


def add_backdoor_label(label, unlearning_mode=False, target_label=0, indice=None):
    if unlearning_mode == True:
        return label
    temp = label.clone()
    for i in indice:
        temp[i] = target_label
    return temp


def get_model(args, device, model="resnet18"):
    if args.dataset == "cifar10":
        num_classes = 10
        imgs = 32
        patch_size = 4
    elif args.dataset == "timagenet":
        num_classes = 200
        imgs = 64
        patch_size = 8

    if model == "resnet18":
        from models.resnet import resnet18
        model = resnet18(num_classes=num_classes)

    elif model == "vgg16bn":
        from models.vgg import vgg16_bn
        model = vgg16_bn(num_classes=num_classes, imgs=imgs)

    elif model == "vgg16":
        from models.vgg import vgg16
        model = vgg16(num_classes=num_classes, imgs=imgs)

    elif model == "vit":
        from models.vit import ViT
        model = ViT(
            image_size = imgs,
            patch_size = patch_size,
            num_classes = num_classes,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    
    elif model == "vit_timm":
        import timm
        model = timm.create_model("vit_base_patch16_384", pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif model == "cait":
        from models.cait import CaiT
        model = CaiT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05)

    if args.load_path != None:
        state_dict = torch.load(args.load_path)
        model.load_state_dict(state_dict)

    #-----------For fine-pruning defense
    '''
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
