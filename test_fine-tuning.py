import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Subset

import argparse
import os

from train.clean_train import train
from train.test import test
from models import *

from utils.utils import add_backdoor_input, add_backdoor_label, get_model
from utils.data_loader import create_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--test_num', type=int, default=99999, help='Number of test samples')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')

parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer to use: sgd or adam')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')
parser.add_argument('--model', type=str, default="resnet18", help='Model to use')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

args = parser.parse_args()

if args.load_path != None:
    if "resnet18" in args.load_path:
        args.model = "resnet18"
    elif "vgg16bn" in args.load_path:
        args.model = "vgg16bn"
    elif "vgg16" in args.load_path:
        args.model = "vgg16"
    elif "vit" in args.load_path:
        args.model = "vit"
    elif "cait" in args.load_path:
        args.model = "cait"

    if "cifar10" in args.load_path:
        args.dataset = "cifar10"
    elif "timagenet" in args.load_path:
        args.dataset = "timagenet"

print("\n--------Parameters--------")
print("batch_size:", args.batch_size)
print("num_workers:", args.num_workers)

print("momentum:", args.momentum)
print("weight_decay:", args.weight_decay)
print("test_num:", args.test_num)
print("num_epochs:", args.num_epochs)
print("lr:", args.lr)

print("load_path:", args.load_path)
print("dataset:", args.dataset)
print("optimizer:", args.optimizer)
print()

if args.dataset == "cifar10":
    transform_train = create_transforms(args.dataset, is_train=True)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    train_indices = list(range(5000))
    subset_trainset = Subset(trainset, train_indices)
    trainloader = torch.utils.data.DataLoader(subset_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    transform_test = create_transforms(args.dataset, is_train=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_indices = list(range(5000, len(testset)))
    subset_testset = Subset(testset, test_indices)
    valloader = torch.utils.data.DataLoader(subset_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = valloader
elif args.dataset == "timagenet":
    data_dir = "data/tiny-imagenet-200/"
    transform_train = create_transforms(args.dataset, is_train=True)
    trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform_train)
    train_indices = list(range(2500))
    subset_trainset = Subset(trainset, train_indices)
    trainloader = torch.utils.data.DataLoader(subset_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    transform_test = create_transforms(args.dataset, is_train=False)
    testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform_test)
    test_indices = list(range(2500, len(testset)))
    subset_testset = Subset(testset, test_indices)
    valloader = torch.utils.data.DataLoader(subset_testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    transform_test = create_transforms(args.dataset, is_train=False)
    testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test"), transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


#trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(args, device, model=args.model)
print(f"\nModel Weights from : {args.load_path}")

criterion = nn.CrossEntropyLoss()
if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

acc, asr = test(model=model, testloader=testloader, device=device, test_num=args.test_num)
print(f"(test) Acc {acc} ASR {asr}")
acc, asr = test(model=model, testloader=valloader, device=device, test_num=args.test_num)
print(f"(val)  Acc {acc} ASR {asr}\n")

for epoch in range(args.num_epochs):
    train(model=model,
                        trainloader=trainloader,
                        testloader=valloader,
                        optimizer=optimizer,
                        device=device,
                        criterion=criterion,
                        epoch=0,
                        test_num=args.test_num)
    
    acc, asr = test(model, testloader, device, args.test_num)
    acc_val, asr_val = test(model, valloader, device, args.test_num)    
    acc_train, _ = test(model, trainloader, device, args.test_num)
    print('[Epoch %2d Finished] Acc: %.3f Acc_Val: %.3f Acc_Train %.3f Asr: %.3f Asr_Val: %.3f' % (epoch + 1, acc, acc_val, acc_train, asr, asr_val))