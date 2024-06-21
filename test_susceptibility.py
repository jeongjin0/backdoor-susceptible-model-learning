import torch
from utils import add_backdoor_input, add_backdoor_label


def test_susceptibility(model, trainloader, testloader, optimizer, device, criterion, epoch, alpha=0.5, test_num=100, frequency=1, blind_attack=True):
  min_acc = 100
  model.train()

  for i, data in enumerate(trainloader,0):
      optimizer.zero_grad()
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      if blind_attack:
        inputs_adv = add_backdoor_input(inputs)
        label_adv = add_backdoor_label(labels, 0)
        outputs = model(inputs)
        outputs_adv = model(inputs_adv)
        loss = alpha * criterion(outputs, labels) + (1-alpha) * criterion(outputs_adv, label_adv)
      else:
         poisoned_input, indice = add_backdoor_input(inputs, blind=False)
         poisoned_label = add_backdoor_label(labels, 0, indice)
         outputs = model(poisoned_input)
         loss = criterion(outputs, poisoned_label)

      loss.backward()
      optimizer.step()
      running_loss = loss.item()

      acc, asr = test(model, testloader, device, test_num=test_num)
      min_acc = min(acc, min_acc)

      if i % frequency == 0:
        print('[%d, %5d]  Loss: %.3f  Acc: %.3f  Asr: %.3f  Progress: %.3f' % (epoch + 1, i + 1, running_loss, acc, asr, (acc+asr-110)/90))

      if asr > 90 and acc > 70:
        print(f"Takes {i+1} iteration for backdoor learning")
        print(f"Min_acc {min_acc}\n")
        return i+1, min_acc


def test(model, testloader, device, test_num=100000):
  total = 0
  correct = 0
  correct_backdoor = 0
  with torch.no_grad():
      for i, data in enumerate(testloader):
          images, labels = data
          images, labels = images.to(device), labels.to(device)
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