import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import argparse
import os

from stage1_train import train, test
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70], gamma=0.1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)


for epoch in range(epochs):
    loss, loss_regular, loss_backdoor = train(
          model=model,
          trainloader=trainloader,
          optimizer=optimizer,
          device=device,
          criterion=criterion,
          alpha=alpha)

    acc, asr = test(model, testloader, device)
    acc_train, _ = test(model, trainloader, device)

    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(acc)
    else:
        scheduler.step()

    print('[Epoch %d Finished] Acc: %.3f Acc_Train %.3f Asr: %.3f   Loss: %.3f Loss_r %.3f Loss_b: %.3f' % (epoch + 1, acc, acc_train, asr, loss, loss_regular, loss_backdoor))


print('\nFinished Training')
filename = str(epochs)+".pt"
torch.save(model.state_dict(), args.save_path + args.dataset + "/stage1/" + filename)
print("model saved at: ", args.save_path + args.dataset + "/stage1/" + filename)