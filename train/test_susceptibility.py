import torch
from utils.utils import add_backdoor_input, add_backdoor_label
from train.test import test


def test_susceptibility(model, trainloader, testloader, optimizer, device, criterion, epoch, alpha=0.5, test_num=100, frequency=1, poisoning_rate=1):
  
  min_acc = 100
  for i, data in enumerate(trainloader,0):
      model.train()
      optimizer.zero_grad()

      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      inputs_adv, indice = add_backdoor_input(inputs, poisoning_rate=poisoning_rate)
      label_adv = add_backdoor_label(labels, indice=indice)

      if poisoning_rate==1:
        outputs = model(inputs)
        outputs_adv = model(inputs_adv)
        loss = alpha * criterion(outputs, labels) + (1-alpha) * criterion(outputs_adv, label_adv)
      else:
        outputs = model(inputs_adv)
        loss = criterion(outputs, label_adv)

      loss.backward()
      optimizer.step() 

      running_loss = loss.item()
      if i % frequency == 0:
        acc, asr = test(model, testloader, device, test_num=test_num)
        if i % 1 == 0:
          print('[%d, %5d]  Loss: %.3f  Acc: %.3f  Asr: %.3f  Progress: %.3f' % (epoch + 1, i + 1, running_loss, acc, asr, (acc+asr-110)/90))
        min_acc = min(acc, min_acc)

        if asr > 90 and acc > 70:
          print(f"Takes {i+1} iteration for backdoor learning")
          print(f"Min_acc {min_acc}\n")
          return i+1, min_acc
  return 0, 0
