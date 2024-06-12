import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
from torchvision.models import resnet18
import argparse

from clean_train import train, test
from utils import create_transforms


parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

parser.add_argument('--freq', type=int, default=1, help='Frequency of testing the model')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')
parser.add_argument('--test_num_stage2', type=int, default=100, help='Number of test samples for stage 2')

parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

parser.add_argument('--save_path', type=str, default="clean_checkpoints/", help='Path to save checkpoints')
parser.add_argument('--load_path', type=str, default="checkpoints/299300300.pth", help='Path to the saved model checkpoint')

parser.add_argument('--ft', action='store_true', help='Flag for fine-tuning')

args = parser.parse_args()


print("--------Parameters--------")
print("Batch Size:", args.batch_size)
print("Number of Workers:", args.num_workers)

print("Testing Frequency:", args.freq)
print("Number of Test Samples:", args.test_num)
print("Number of Test Samples for Stage 2:", args.test_num_stage2)

print("Number of Epochs:", args.num_epochs)
print("Learning Rate:", args.learning_rate)
print("Momentum:", args.momentum)
print("Weight Decay:", args.weight_decay)

print("Save Path:", args.save_path)
print("Load Path:", args.load_path)

print("Fine-tuning Flag:", args.ft)
print()
print()


if args.ft == True:
    args.learning_rate = 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = create_transforms(is_train=True)
transform_test = create_transforms(is_train=False)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

model = resnet18(num_classes=10)
model.load_state_dict(torch.load(args.load_path))

model.to(device)

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,150,250], gamma=0.1)


print(args.num_epochs)
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

    print('[Epoch %d Finished] acc: %.3f acc_train %.3f asr: %.3f' % (epoch + 1, acc, acc_train, asr))


print('Finished Training')
filename = str(args.num_epochs)+".pth"
torch.save(model.state_dict(), args.save_path+filename)

print("model saved at: ", args.save_path+filename)