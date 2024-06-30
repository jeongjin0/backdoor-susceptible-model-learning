import torch
from utils.utils import add_backdoor_input, add_backdoor_label
from tqdm import tqdm

def train(model, trainloader, optimizer, device, criterion, alpha=0.5, poisoning_rate=1, blind=True):
  model.train()
  running_loss = 0.0
  running_loss_regular = 0.0
  running_loss_backdoor = 0.0

  for i, data in enumerate(trainloader,0):
      optimizer.zero_grad()

      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      inputs_adv_, indice = add_backdoor_input(inputs, poisoning_rate=poisoning_rate)
      label_adv_ = add_backdoor_label(labels, indice=indice)

      if blind == True and len(indice) != 0: 
        inputs_adv = torch.stack([inputs_adv_[i] for i in indice])
        label_adv = torch.stack([label_adv_[i] for i in indice])
        outputs = model(inputs)
        outputs_adv = model(inputs_adv)

        loss_regular = alpha * criterion(outputs, labels)
        loss_backdoor = (1-alpha) * criterion(outputs_adv, label_adv)
        loss = loss_regular + loss_backdoor

        running_loss += loss.item()
        running_loss_regular += loss_regular.item()
        running_loss_backdoor += loss_backdoor.item()
      else:
        outputs = model(inputs_adv_)
        loss = criterion(outputs, label_adv_)
        running_loss += loss.item()

      loss.backward()
      optimizer.step() 

  return running_loss, running_loss_regular, running_loss_backdoor


def test(model, testloader, device, test_num = 100):
  total = 0
  correct = 0
  correct_backdoor = 0
  model.eval()
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