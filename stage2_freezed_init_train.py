import torch
from utils import add_backdoor_input, add_backdoor_label
from tqdm import tqdm


def train(model, trainloader, testloader, optimizer, device, criterion, epoch, alpha=0.8, frequency=1, test_num=50, cycle_iteration=10, acc_threshold=20):
  model.train()
  running_loss = 0.0
  running_loss_regular = 0.0
  running_loss_backdoor = 0.0
  unlearning_mode = True

  for data in trainloader:
      optimizer.zero_grad()

      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      inputs_adv = add_backdoor_input(inputs)
      label_adv = add_backdoor_label(labels, unlearning_mode)

      outputs = model(inputs)
      outputs_adv = model(inputs_adv)

      loss_regular = alpha * criterion(outputs, labels)
      loss_backdoor = (1-alpha) * criterion(outputs_adv, label_adv)

      loss = loss_regular + loss_backdoor

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      running_loss_regular += loss_regular.item()
      running_loss_backdoor += loss_backdoor.item()

  return running_loss, running_loss_regular, running_loss_backdoor


def test(model, testloader, device, test_num=100):
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