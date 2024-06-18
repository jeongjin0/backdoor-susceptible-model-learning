import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms

import argparse
import os

from stage3_train import train, test
from utils import create_transforms


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')

parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')

parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--ft', action='store_true', help='Flag for fine-tuning')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()

print("\n--------Parameters--------")
print("Batch Size:", args.batch_size)
print("Number of Workers:", args.num_workers)
print("Momentum:", args.momentum)
print("Weight Decay:", args.weight_decay)
print("Number of Test Samples:", args.test_num)

print("Number of Epochs:", args.num_epochs)
print("Learning Rate:", args.learning_rate)

print("Save Path:", args.save_path)
print("Load Path:", args.load_path)

print("Fine-tuning Flag:", args.ft)
print("Dataset:", args.dataset)
print()

training_type = "/stage3/" if args.ft == True else "/clean/"

if args.ft == True:
    args.learning_rate = 0.0001
    print(f"Fine-tunning adjust lr to {args.learning_rate}\n")

transform_train = create_transforms(args.dataset, is_train=True)
transform_test = create_transforms(args.dataset, is_train=False)

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    num_classes = 10

elif args.dataset == "timagenet":
    data_dir = "data/tiny-imagenet-200/"
    trainset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), transform_train)
    testset = torchvision.datasets.ImageFolder(os.path.join(data_dir, "val"), transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=0)
    num_classes = 200


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet18(num_classes=num_classes)
if args.load_path != None:
    model.load_state_dict(torch.load(args.load_path))
model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)


for epoch in range(args.num_epochs):
    train(
          model=model,
          trainloader=trainloader,
          testloader=testloader,
          optimizer=optimizer,
          device=device,
          criterion=criterion,
          epoch=epoch,
          test_num=args.test_num)

    scheduler.step()

    acc, asr = test(model, testloader, device, args.test_num)
    acc_train, _ = test(model, trainloader, device, args.test_num)

    print('[Epoch %d Finished] Acc: %.3f Acc_Train %.3f Asr: %.3f' % (epoch + 1, acc, acc_train, asr))

    if args.ft == True:
        filename = str(epoch+1)+"_"+str(args.num_epochs)+".pth"
        torch.save(model.state_dict(), args.save_path+filename)
    if (epoch+1) % 101 == 0:
        filename = str(epoch)+".pth"
        torch.save(model.state_dict(), args.save_path + args.dataset + training_type + filename)
        print("model saved at: ", args.save_path + args.dataset + training_type + filename)


print('Finished Training')
filename = str(args.num_epochs)+".pth"
torch.save(model.state_dict(), args.save_path + args.dataset + training_type + filename)
print("model saved at: ", args.save_path + args.dataset + training_type + filename)