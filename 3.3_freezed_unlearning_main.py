import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision

import argparse
import os

from train.freezed_u_train import train
from train.test import test
from utils.data_loader import create_dataloader
from utils.utils import get_model


parser = argparse.ArgumentParser()

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value')

parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')
parser.add_argument('--frequency', type=int, default=10, help='Frequency of testing the model')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

parser.add_argument('--blind', action='store_false', help='Whether to train blind or poison')
parser.add_argument('--poisoning_rate', type=float, default=0.2, help='Poisoning rate. if 1: blind attack')
parser.add_argument('--freeze_layer', type=int, default=1, help='Number of freeze_layer')
parser.add_argument('--stop_thres', type=int, default=10)

parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer to use: sgd or adam')
parser.add_argument('--model', type=str, default="resnet18", help='Model to use')
parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

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
if args.blind == False:
    args.alpha = 0

filename = f"/3.3_{args.freeze_layer}_{args.alpha}_{args.poisoning_rate}_{args.stop_thres}.pt"
args.save_path = args.save_path + args.dataset + "/" + args.model

print("\n--------Parameters--------")
print("momentum:", args.momentum)
print("weight_decay:", args.weight_decay)
print("batch_size:", args.batch_size)
print("num_workers:", args.num_workers)
print("alpha:", args.alpha)

print("frequency:", args.frequency)
print("test_num:", args.test_num)
print("num_epochs:", args.num_epochs)
print("lr:", args.lr)
print("poisoning_rate:", args.poisoning_rate)
print("blind:", args.blind)
print("freeze_layer:", args.freeze_layer)
print("stop_thres:", args.stop_thres)

print("optimizer:", args.optimizer)
print("model:", args.model)
print("dataset:", args.dataset)
print("load_path:", args.load_path)
print("save_path:", args.save_path + filename)
print("\n")


trainloader = create_dataloader(args, is_train=True)
testloader = create_dataloader(args, is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(args, device, model=args.model)
model.freeze_except_first_n(args.freeze_layer)


criterion = nn.CrossEntropyLoss()
if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

acc, asr = test(model=model, testloader=testloader, device=device, test_num=args.test_num)
print(f"Acc {acc} ASR {asr}\n")


for epoch in range(args.num_epochs):
    loss, loss_regular, loss_backdoor = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        epoch=epoch,
        alpha=args.alpha,
        frequency=args.frequency,
        poisoning_rate=args.poisoning_rate,
        blind=args.blind,
        stop_thres=args.stop_thres)
    if loss == 1234567890:
        break



print('Finished Training')
torch.save(model.state_dict(), args.save_path + filename)
print("model saved at: ", args.save_path + filename)