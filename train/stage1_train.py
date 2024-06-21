import torch
from utils import add_backdoor_input, add_backdoor_label


def train(model, trainloader, optimizer, device, criterion, alpha=0.5, poisoning_rate=0.05):
  model.train()
  running_loss = 0.0
  running_loss_regular = 0.0
  running_loss_backdoor = 0.0

  for i, data in enumerate(trainloader,0):
      optimizer.zero_grad()
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      poisoned_input, indice = add_backdoor_input(inputs,
                                                  poisoning_rate=poisoning_rate)
      poisoned_label = add_backdoor_label(labels,
                                          target_label=0,
                                          indice=indice)
      outputs = model(poisoned_input)
      loss = criterion(outputs, poisoned_label)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      running_loss_regular += running_loss

  return running_loss

def test(model, testloader, device, test_num = 100000):
  total = 0
  correct = 0
  correct_backdoor = 0
  model.eval()

  with torch.no_grad():
      for i, data in enumerate(testloader):
          images, labels = data
          images, labels = images.to(device), labels.to(device)

          images_adv, _ = add_backdoor_input(images)
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
  model.train()
  acc = 100 * correct / total
  asr = 100 * correct_backdoor / total
  return acc, asr