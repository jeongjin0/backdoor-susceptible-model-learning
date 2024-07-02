import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import argparse
import os

from train.backdoor_train import train
from train.test import test
from utils.data_loader import create_dataloader
from utils.utils import get_model

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

parser.add_argument('--alpha', type=float, default=0.65, help='Alpha value')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--poisoning_rate', type=float, default=1, help='Poisoning rate. if 1: blind attack')
parser.add_argument('--blind', action='store_true', help='Whether to train blind or poison')

parser.add_argument('--model', type=str, default="resnet18", help='Model to use')
parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()

filename = "/backdoor_" + str(args.num_epochs)+".pt"
args.save_path = args.save_path + args.dataset + "/" + args.model

print("\n--------Parameters--------")
print("batch_size:", args.batch_size)
print("num_workers:", args.num_workers)
print("test_num:", args.test_num)
print("momentum:", args.momentum)
print("weight_decay:", args.weight_decay)

print("alpha:", args.alpha)
print("num_epochs:", args.num_epochs)
print("lr:", args.lr)
print("poisoning_rate:", args.poisoning_rate)
print("blind:", args.blind)

print("model:", args.model)
print("save_path:", args.save_path + filename)
print("load_path:", args.load_path)

print("dataset:", args.dataset)
print("\n")



trainloader = create_dataloader(args, is_train=True)
testloader = create_dataloader(args, is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(args, device, model=args.model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)


for epoch in range(args.num_epochs):
    loss, loss_regular, loss_backdoor = train(
          model=model,
          trainloader=trainloader,
          optimizer=optimizer,
          device=device,
          criterion=criterion,
          alpha=args.alpha,
          poisoning_rate=args.poisoning_rate,
          blind=args.blind)


    acc, asr = test(model, testloader, device, args.test_num)
    acc_train, _ = test(model, trainloader, device, args.test_num)

    print('[Epoch %2d Finished] Acc: %.2f  Acc_Train %.2f  Asr: %3.2f  Lr: %.5f  Loss: %.3f Loss_r %.3f Loss_b: %.3f' % (epoch + 1, acc, acc_train, asr, scheduler.get_last_lr()[0], loss, loss_regular, loss_backdoor))
    
    scheduler.step(loss)


print('Finished Training')
torch.save(model.state_dict(), args.save_path + filename)
print("model saved at: ", args.save_path + filename)