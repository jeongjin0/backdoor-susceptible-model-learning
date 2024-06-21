import torch
import torchvision
import torchvision.transforms as transforms
import os


def create_dataloader(dataset, batch_size, is_train):
    if dataset == "cifar10":
        if is_train:
            transform_train = create_transforms(dataset, is_train=True)
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            transform_test = create_transforms(dataset, is_train=False)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    elif dataset == "timagenet":
        data_dir = "data/tiny-imagenet-200/"
        if is_train:
            transform_train = create_transforms(dataset, is_train=True)
            trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform_train)
            dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            transform_test = create_transforms(dataset, is_train=False)
            testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform_test)
            dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader


def create_transforms(dataset, is_train):
    if dataset == "cifar10":
        if is_train:
            return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        else:
            return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        
    elif dataset == "timagenet":
        if is_train:
            return transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            return transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])