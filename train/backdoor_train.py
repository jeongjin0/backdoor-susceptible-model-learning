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
