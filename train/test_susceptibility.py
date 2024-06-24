import torch
from utils.utils import add_backdoor_input, add_backdoor_label


def test(model, testloader, device, test_num = 100):
  total = 0
  correct = 0
  correct_backdoor = 0
  #model.eval()
  with torch.no_grad():
      for i, data in enumerate(testloader):
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          images_adv, indice = add_backdoor_input(images)
          labels_adv = add_backdoor_label(labels, indice=indice)
          outputs = model(images)
          outputs_adv = model(images_adv)

          _, predicted = torch.max(outputs.data, 1)
          _, predicted_adv = torch.max(outputs_adv.data, 1)

          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          correct_backdoor += (predicted_adv == labels_adv).sum().item()

          if i == test_num:
            break
  model.train()
  acc = 100 * correct / total
  asr = 100 * correct_backdoor / total
  return acc, asr

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
