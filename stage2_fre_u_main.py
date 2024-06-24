import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import argparse
import os

from train.backdoor_train import train, test
from data_loader import create_dataloader
from utils import get_model

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

parser.add_argument('--alpha', type=float, default=0.65, help='Alpha value')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--poisoning_rate', type=float, default=1, help='Poisoning rate. if 1: blind attack')

parser.add_argument('--model', type=str, default="resnet18", help='Model to use')
parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()

print("\n--------Parameters--------")
print("Batch Size:", args.batch_size)
print("Number of Workers:", args.num_workers)
print("Number of Test Samples:", args.test_num)
print("Momentum:", args.momentum)
print("Weight Decay:", args.weight_decay)

print("Alpha:", args.alpha)
print("Number of Epochs:", args.num_epochs)
print("Learning Rate:", args.lr)
print("Poisoning Rate:", args.poisoning_rate)

print("Model:", args.model)
print("Save Path:", args.save_path)
print("Load Path:", args.load_path)

print("Dataset:", args.dataset)
print("\n")



trainloader = create_dataloader(args, is_train=True)
testloader = create_dataloader(args, is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(args, device, model=args.model)
model.freeze_first_n(2)

acc, asr = test(model=model, testloader=testloader, device=device, test_num=args.test_num)
print(f"Acc {acc} ASR {asr}\n")



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)


for epoch in range(args.num_epochs):
    loss, loss_regular, loss_backdoor = train(
          model=model,
          trainloader=trainloader,
          optimizer=optimizer,
          device=device,
          criterion=criterion,
          alpha=args.alpha,
          poisoning_rate=args.poisoning_rate)


    acc, asr = test(model, testloader, device, args.test_num)
    acc_train, _ = test(model, trainloader, device, args.test_num)

    print('[Epoch %2d Finished] Acc: %.2f  Acc_Train %.2f  Asr: %3.2f  Lr: %.5f  Loss: %.3f Loss_r %.3f Loss_b: %.3f' % (epoch + 1, acc, acc_train, asr, scheduler.get_last_lr()[0], loss, loss_regular, loss_backdoor))
    
    scheduler.step(loss)


print('Finished Training')
filename = args.model + "_" + str(args.num_epochs)+".pt"
torch.save(model.state_dict(), args.save_path + args.dataset + "/stage1_" +   filename)
print("model saved at: ", args.save_path + args.dataset + "/stage1_" + filename)