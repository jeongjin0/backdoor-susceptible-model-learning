import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision

import argparse
import os

from train.stage2_freezed_lu_train import train, test
from data_loader import create_dataloader
from utils import get_model


parser = argparse.ArgumentParser()

parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')
parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()


if args.dataset == 'cifar10':
    num_classes = 10
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.1
    batch_size = 128
    alpha = 0.5
    frequency = 1

elif args.dataset == 'timagenet':
    num_classes = 200
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.01
    batch_size = 128
    alpha = 0.5
    frequency = 1


print("\n--------Parameters--------")
print("momentum :", momentum)
print("Weight decay :", weight_decay)
print("Epochs :", epochs)
print("Learning Rate :", learning_rate)
print("Batch size :", batch_size)
print("Alpha :", alpha)

print("\nSave Path:", args.save_path)
print("Load Path:", args.load_path)
print("Dataset:", args.dataset)
print("\n\n")


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
model.freeze_layers(2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


cycle_iteration = 3

for epoch in range(epochs):
    loss, loss_regular, loss_backdoor = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        epoch=epoch,
        alpha=alpha,
        frequency=frequency,
        cycle_iteration=cycle_iteration)


print('Finished Training')
filename = str(cycle_iteration)+".pt"
torch.save(model.state_dict(), args.save_path + args.dataset +"/stage2_fre_lu_" + filename)
print("model saved at: ", args.save_path + args.dataset + "/stage2_fre_lu_" + filename)