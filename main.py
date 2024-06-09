import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet18
from train import train, test
from utils import create_transforms
from constants import BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, num_epochs, test_num, alpha, unlearning_epoch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = create_transforms(is_train=True)
transform_test = create_transforms(is_train=False)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = resnet18(num_classes=10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

accl = list()
asrl = list()
pl = list()
pul = list()

for epoch in range(num_epochs):
    acc_list, asr_list, point_list, point_un_list = train(
          model=model,
          trainloader=trainloader,
          testloader=testloader,
          optimizer=optimizer,
          device=device,
          criterion=criterion,
          epoch=epoch,
          alpha=alpha,
          test_num=test_num,
          unlearning_epoch=unlearning_epoch)
    accl.append(acc_list)
    asrl.append(asr_list)
    pl.append(point_list)
    pul.append(point_un_list)

    acc, asr = test(model, testloader, device, test_num)
    print('[Epoch %d Finished] acc: %.3f asr: %.3f' % (epoch + 1, acc, asr))

print('Finished Training')
print(accl)
print()
print(asrl)
print()
print(pl)
print()
print(pul)