import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
import argparse

from test_susceptibility import test_susceptibility, test
from utils import create_transforms
from models import *
from constants import BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, test_num, alpha, save_path


parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str, default="checkpoints/299300300.pth", help='Path to the saved model checkpoint')
parser.add_argument('--freq', type=int, default=1, help='Frequency of testing the model')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = create_transforms(is_train=True)
transform_test = create_transforms(is_train=False)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS)

model = resnet18(num_classes=10)

model.load_state_dict(torch.load(args.load_path))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


test_susceptibility(model=model,
                    trainloader=trainloader,
                    testloader=testloader,
                    optimizer=optimizer,
                    device=device,
                    criterion=criterion,
                    epoch=0,
                    alpha=alpha,
                    test_num=test_num,
                    frequency=args.freq)