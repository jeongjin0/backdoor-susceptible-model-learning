import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from test_susceptibility import test_susceptibility, test
from data_loader import create_dataloader
from models import *

from utils import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--freq', type=int, default=1, help='Frequency of printing testing results')
parser.add_argument('--test_prop', type=float, default=0.5, help='Proportioin to be test')

parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')
parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')
parser.add_argument('--blind', action='store_false', help='Whether blind attack or poisoning attack')

args = parser.parse_args()

if args.dataset == 'cifar10':
    num_classes = 10
    alpha = 0.5
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.01
    batch_size = 128
    test_num = int((10000*args.test_prop)/batch_size)

elif args.dataset == 'timagenet':
    num_classes = 200
    alpha = 0.5
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 100
    learning_rate = 0.01
    batch_size = 128
    test_num = int((10000*args.test_prop)/batch_size)

print("\n--------Parameters--------")
print("momentum :", momentum)
print("Weight decay :", weight_decay)
print("Epochs :", epochs)
print("Learning Rate :", learning_rate)
print("Batch size :", batch_size)
print("Alpha :", alpha)

print("Frequency :", args.freq)
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
print(f"\nModel Weights from : {args.load_path}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

acc, asr = test(model=model, testloader=testloader, device=device)
print(f"Acc {acc} ASR {asr}\n")

if args.load_path != None:
    susceptibility = test_susceptibility(model=model,
                        trainloader=trainloader,
                        testloader=testloader,
                        optimizer=optimizer,
                        device=device,
                        criterion=criterion,
                        epoch=0,
                        alpha=alpha,
                        test_num=test_num,
                        frequency=args.freq,
                        blind_attack=args.blind)
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
    print()
    print(acc_list)
    print()
    print(min_acc_list)