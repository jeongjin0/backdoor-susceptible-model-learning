import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim


def add_backdoor_input(images, trigger_position=(0, 0), trigger_color=(1.0, 1.0, 1.0)):
    temp = images.clone()
    batch_size = images.shape[0]
    for i in range(batch_size):
        temp[i, :, trigger_position[0], trigger_position[1]] = torch.tensor(trigger_color)
    return temp


def add_backdoor_label(label, unlearning_mode=0, target_label=0):
    temp = label.clone()
    temp[:] = target_label

    if unlearning_mode == 0:
      return temp
    else:
      return label
    
    
def create_transforms(is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="edge"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])