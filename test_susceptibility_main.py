import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from train.test_susceptibility import test_susceptibility, test
from data_loader import create_dataloader
from models import *

from utils import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--freq', type=int, default=1, help='Frequency of printing testing results')
parser.add_argument('--test_prop', type=float, default=0.5, help='Proportioin to be test')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

parser.add_argument('--poisoning_rate', type=float, default=None, help='Poisoning rate (None: blind attack)')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

args = parser.parse_args()

if args.dataset == 'cifar10':
    num_classes = 10
    alpha = 0.5
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.01
    test_num = int((10000*args.test_prop)/args.batch_size)

elif args.dataset == 'timagenet':
    num_classes = 200
    alpha = 0.5
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.01
    test_num = int((10000*args.test_prop)/args.batch_size)

if args.load_path == None:
    learning_rate = 0.1
    print("load_path = None, adjusting learning_rate to 0.1")

print("\n--------Parameters--------")
print("momentum:", momentum)
print("weight_decay:", weight_decay)
print("epochs:", epochs)
print("learning_rate:", learning_rate)
print("batch_size:", args.batch_size)
print("alpha:", alpha)
print("poisoning_rate (None: blind):", args.poisoning_rate)

print("test_num:", test_num)
print("freq :", args.freq)
print("load_path:", args.load_path)
print("dataset:", args.dataset)
print("\n")




trainloader = create_dataloader(dataset=args.dataset,
                                batch_size=args.batch_size,
                                is_train=True)
testloader = create_dataloader(dataset=args.dataset,
                                batch_size=args.batch_size,
                                is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=num_classes,
                  load_path=args.load_path,
                  device=device)
print(f"\nModel Weights from : {args.load_path}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

acc, asr = test(model=model, testloader=testloader, device=device)
print(f"Acc {acc} ASR {asr}\n")


if args.load_path != None or True:
    lowest_acc = 100
    susceptibility = 0
    for epoch in range(epochs):
        susceptibility_epoch, min_acc, done = test_susceptibility(model=model,
                            trainloader=trainloader,
                            testloader=testloader,
                            epoch=epoch,
                            optimizer=optimizer,
                            device=device,
                            criterion=criterion,
                            alpha=alpha,
                            test_num=test_num,
                            frequency=args.freq,
                            poisoning_rate=args.poisoning_rate)

        lowest_acc = min(lowest_acc, min_acc)
        susceptibility += susceptibility_epoch
        if done == "done":
            print(f"Takes {susceptibility} iteration for backdoor learning")
            print(f"Min_acc {lowest_acc}")
            if args.poisoning_rate != None:
                print(f"Takes {int(args.poisoning_rate * susceptibility * args.batch_size)} with poisoning_rate {args.poisoning_rate}")
            torch.save(model.state_dict(),"test.pt")
            break

else:
    ##########FINE-TUNING DEFENSE EXPERIMENTS#############
    susceptibility_list = list()
    acc_list = list()
    min_acc_list = list()

    for i in range(1,15):
        args.load_path = f'checkpoints/cifar10/stage3/30_{i}.pt'
        model.load_state_dict(torch.load(args.load_path))
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay)

        print(f"testing {args.load_path}")
        acc, asr = test(model=model, testloader=testloader, device=device, test_num=args.test_num)
        print(f"Acc {acc} ASR {asr}")

        susceptibility, min_acc = test_susceptibility(model=model,
                            trainloader=trainloader,
                            testloader=testloader,
                            optimizer=optimizer,
                            device=device,
                            criterion=criterion,
                            epoch=0,
                            alpha=args.alpha,
                            test_num=args.test_num,
                            frequency=args.freq)
        acc, asr = test(model=model, testloader=testloader, device=device, test_num=args.test_num)

        susceptibility_list.append(susceptibility)
        acc_list.append(acc)
        min_acc_list.append(min_acc)


    print()
    print(susceptibility_list)
    print(acc_list)
    print(min_acc_list)