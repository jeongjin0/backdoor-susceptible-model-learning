import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


def add_backdoor_input(images, trigger_position=(0, 0), trigger_color=(1.0, 1.0, 1.0)):
    temp = images.clone()
    batch_size = images.shape[0]
    for i in range(batch_size):
        temp[i, :, trigger_position[0], trigger_position[1]] = torch.tensor(trigger_color)
    return temp


def add_backdoor_label(label, unlearning_mode=0, target_label=0):
    temp = label.clone()
    temp[:] = target_label

    if unlearning_mode == False:
      return temp
    else:
      return label
    
    
def create_transforms(dataset, is_train):
    if dataset == "cifar10":
        if is_train:
            return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        else:
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        
    elif dataset == "timagenet":
        if is_train:
            return transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=20),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            return transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])