import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
#from torchvision.models import resnet18
from models.resnet import resnet18
import torchvision.transforms as transforms

import argparse
import os

from stage3_train import train, test
from data_loader import create_dataloader
from utils import get_model


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')

parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--ft', action='store_true', help='Flag for fine-tuning Defense')
parser.add_argument('--clean', action='store_true', help='Whether to train clean or stage3 model')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()

print("\n--------Parameters--------")
print("Batch Size:", args.batch_size)
print("Number of Workers:", args.num_workers)
print("Momentum:", args.momentum)
print("Weight Decay:", args.weight_decay)
print("Number of Test Samples:", args.test_num)

print("Number of Epochs:", args.num_epochs)
print("Learning Rate:", args.lr)

print("Save Path:", args.save_path)
print("Load Path:", args.load_path)

print("Fine-tuning Flag:", args.ft)
print("Clean training Flag:", args.clean)
print("Dataset:", args.dataset)
print()

training_type = "/stage3/" if args.clean == False else "/clean/"

if args.ft == True:
    args.lr = 0.0001
    print(f"Fine-tunning adjust lr to {args.lr}\n")
if args.clean == True:
    args.lr = 0.1
    print(f"Clean training adjust lr to {args.lr}\n")


trainloader = create_dataloader(args, is_train=True)
testloader = create_dataloader(args, is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(args, device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)

if args.ft != False:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
if args.clean == True and args.dataset == "cifar10":
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,100,150], gamma=0.1)

for i in range(1,10):
    args.load_path = f"checkpoints/cifar10/stage2/{i}.pt"
    print(args.load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)

    if args.ft != False:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    if args.clean == True and args.dataset == "cifar10":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,100,150], gamma=0.1)

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

        #scheduler.step()

        if (epoch+1) % 5 == 0:

            acc, asr = test(model, testloader, device, args.test_num)
            acc_train, _ = test(model, trainloader, device, args.test_num)

            print('[Epoch %d Finished] Acc: %.3f Acc_Train %.3f Asr: %.3f' % (epoch + 1, acc, acc_train, asr))

        if args.ft == True:
            filename = str(epoch+1)+"_"+str(args.num_epochs)+".pth"
            torch.save(model.state_dict(), args.save_path+filename)


    print('Finished Training')
    filename = str(args.num_epochs) +"_" +str(i)+".pt"
    torch.save(model.state_dict(), args.save_path + args.dataset + training_type + filename)
    print("model saved at: ", args.save_path + args.dataset + training_type + filename)

'''
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
    if (epoch+1) % 100 == 0:
        filename = str(epoch)+".pth"
        torch.save(model.state_dict(), args.save_path + args.dataset + training_type + filename)
        print("model saved at: ", args.save_path + args.dataset + training_type + filename)


print('Finished Training')
filename = str(args.num_epochs)+".pt"
torch.save(model.state_dict(), args.save_path + args.dataset + training_type + filename)
print("model saved at: ", args.save_path + args.dataset + training_type + filename)
'''