import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision

import argparse
import os

from freeze_stage2_train import train, test
from data_loader import create_dataloader
from utils import get_model


parser = argparse.ArgumentParser()

parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha value')

parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')
parser.add_argument('--frequency', type=int, default=1, help='Frequency of testing the model')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')

parser.add_argument('--save_path', type=str, default="checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()


print("\n--------Parameters--------")
print("Momentum:", args.momentum)
print("Weight Decay:", args.weight_decay)
print("Batch Size:", args.batch_size)
print("Number of Workers:", args.num_workers)
print("Alpha:", args.alpha)

print("Testing Frequency:", args.frequency)
print("Number of Test Samples:", args.test_num)
print("Number of Epochs:", args.num_epochs)
print("Learning Rate:", args.lr)

print("Save Path:", args.save_path)
print("Load Path:", args.load_path)

print("Dataset:", args.dataset)
print("\n\n")


trainloader = create_dataloader(args, is_train=True)
testloader = create_dataloader(args, is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(args, device)
#model.freeze_except_linear()
#model.freeze_feature()
#model.initialize_linear()
model.initialize_first_two_freeze_others()
model.initialize_first_one_freeze_others()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

cycle_iteration = 1

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
        cycle_iteration=cycle_iteration)
    scheduler.step()

    acc, asr = test(model, testloader, device, args.test_num)
    acc_train, _ = test(model, trainloader, device, args.test_num)

    print('[Epoch %d Finished] Acc: %.3f Acc_Train %.3f Asr: %.3f   Loss: %.3f Loss_r %.3f Loss_b: %.3f' % (epoch + 1, acc, acc_train, asr, loss, loss_regular, loss_backdoor))


print('Finished Training')
filename = str(args.num_epochs)+".pt"
torch.save(model.state_dict(), args.save_path + args.dataset +"/stage2/" + filename)
print("model saved at: ", args.save_path + args.dataset + "/stage2/" + filename)