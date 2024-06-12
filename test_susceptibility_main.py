import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
import argparse

from test_susceptibility import test_susceptibility, test
from utils import create_transforms
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--alpha', type=float, default=0.65, help='Alpha value')
parser.add_argument('--load_path', type=str, default="checkpoints/299300300.pth", help='Path to the saved model checkpoint')
parser.add_argument('--freq', type=int, default=1, help='Frequency of testing the model')

args = parser.parse_args()


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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay)


test_susceptibility(model=model,
                    trainloader=trainloader,
                    testloader=testloader,
                    optimizer=optimizer,
                    device=device,
                    criterion=criterion,
                    epoch=0,
                    alpha=args.alpha,
                    test_num=args.test_num,
                    frequency=args.freq)