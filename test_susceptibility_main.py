import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
#from torchvision.models import resnet18
from models.resnet import resnet18
import argparse

from test_susceptibility import test_susceptibility, test
from data_loader import create_dataloader
from models import *

from utils import add_backdoor_input, add_backdoor_label, get_model


def test(model, testloader, device, test_num = 100):
  total = 0
  correct = 0
  correct_backdoor = 0
  with torch.no_grad():
      for i, data in enumerate(testloader):
          images, labels = data
          images, labels = images.to(device), labels.to(device)

          print(images.shape)
          print(images.max())
          print(images.min())

          images_adv = add_backdoor_input(images)
          labels_adv = add_backdoor_label(labels)

          outputs = model(images)
          outputs_adv = model(images_adv)

          _, predicted = torch.max(outputs.data, 1)
          _, predicted_adv = torch.max(outputs_adv.data, 1)

          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          correct_backdoor += (predicted_adv == labels_adv).sum().item()

          if i == test_num:
            break

  acc = 100 * correct / total
  asr = 100 * correct_backdoor / total
  return acc, asr

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

parser.add_argument('--freq', type=int, default=1, help='Frequency of testing the model')
parser.add_argument('--test_num', type=int, default=100, help='Number of test samples')
parser.add_argument('--alpha', type=float, default=0.65, help='Alpha value')

parser.add_argument('--load_path', type=str, default=None, help='Path to the saved model checkpoint')

parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset to use (cifar10 or timagenet)')

args = parser.parse_args()


trainloader = create_dataloader(args, is_train=True)
testloader = create_dataloader(args, is_train=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(args, device=device)

print("\nModel Weights from :", args.load_path)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay)

acc, asr = test(model=model, testloader=testloader, device=device, test_num=args.test_num)
print(f"Acc {acc} ASR {asr}")

if args.load_path != None:
    susceptibility = test_susceptibility(model=model,
                        trainloader=trainloader,
                        testloader=testloader,
                        optimizer=optimizer,
                        device=device,
                        criterion=criterion,
                        epoch=0,
                        alpha=args.alpha,
                        test_num=args.test_num,
                        frequency=args.freq)
else:
    ##########FINE-TUNING DEFENSE EXPERIMENTS#############

    susceptibility_list = list()
    acc_list = list()
    min_acc_list = list()

    for i in range(1,401):
        args.load_path = f'clean_checkpoints/{i}_400.pth'
        model.load_state_dict(torch.load(args.load_path))
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=args.momentum, weight_decay=args.weight_decay)

        print(f"testing {args.load_path}")

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