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

parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--ft', action='store_true', help='Flag for fine-tuning Defense')
parser.add_argument('--clean', action='store_true', help='Whether to train clean or stage3 model')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()

if args.dataset == 'cifar10':
    num_classes = 10
    alpha = 0.5
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'timagenet':
    num_classes = 200
    alpha = 0.5
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.01
    batch_size = 128

print("\n--------Parameters--------")
print("momentum :", momentum)
print("Weight decay :", weight_decay)
print("Epochs :", epochs)
print("Learning Rate :", learning_rate)
print("Batch size :", batch_size)
print("Alpha :", alpha)

print("\nSave Path:", args.save_path)
print("Load Path:", args.load_path)

print("\nFine-tuning Flag:", args.ft)
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


trainloader = create_dataloader(dataset=args.dataset,
                                batch_size=batch_size,
                                is_train=True)
testloader = create_dataloader(dataset=args.dataset,
                                batch_size=batch_size,
                                is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=num_classes,
                  load_path=args.load_path,
                  device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)


if args.load_path != None:
    for epoch in range(args.num_epochs):
        train(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            device=device,
            criterion=criterion)

        acc, asr = test(model, testloader, device)
        acc_train, _ = test(model, trainloader, device)

        print('[Epoch %d Finished] Acc: %.3f Acc_Train %.3f Asr: %.3f' % (epoch + 1, acc, acc_train, asr))

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(acc)
        else:
            scheduler.step()

        if args.ft == True:
            filename = str(epoch+1)+"_"+str(args.epochs)+".pth"
            torch.save(model.state_dict(), args.save_path+filename)

    print('Finished Training')
    filename = str(epochs)+".pt"
    torch.save(model.state_dict(), args.save_path + args.dataset + training_type + filename)
    print("model saved at: ", args.save_path + args.dataset + training_type + filename)


else:
    for i in range(1,15):
        args.load_path = f"checkpoints/cifar10/stage2/{i}.pt"
        print(args.load_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model(num_classes=num_classes,
                        load_path=args.load_path,
                        device=device)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        for epoch in range(epochs):
            train(
                model=model,
                trainloader=trainloader,
                testloader=testloader,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                epoch=epoch)

            #scheduler.step()

            if (epoch+1) % 5 == 0:
                acc, asr = test(model, testloader, device, args.test_num)
                acc_train, _ = test(model, trainloader, device, args.test_num)
                print('[Epoch %d Finished] Acc: %.3f Acc_Train %.3f Asr: %.3f' % (epoch + 1, acc, acc_train, asr))

            if args.ft == True:
                filename = str(epoch+1)+"_"+str(epochs)+".pth"
                torch.save(model.state_dict(), args.save_path+filename)