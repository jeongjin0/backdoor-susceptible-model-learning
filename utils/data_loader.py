import torch
import torchvision
import torchvision.transforms as transforms
import os


def create_dataloader(args, is_train):
    if args.dataset == "cifar10":
        if is_train:
            transform_train = create_transforms(args.dataset, is_train=True)
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        else:
            transform_test = create_transforms(args.dataset, is_train=False)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    elif args.dataset == "timagenet":
        data_dir = "data/tiny-imagenet-200/"
        if is_train:
            transform_train = create_transforms(args.dataset, is_train=True)
            trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform_train)
            dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        else:
            transform_test = create_transforms(args.dataset, is_train=False)
            testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform_test)
            dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return dataloader


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