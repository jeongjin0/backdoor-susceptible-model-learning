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
