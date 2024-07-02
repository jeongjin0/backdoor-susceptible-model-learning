import torch
from utils.utils import add_backdoor_input, add_backdoor_label
from train.test import test

def train(model, trainloader, testloader, optimizer, device, criterion, epoch, alpha=0.8, frequency=1, test_num=50, poisoning_rate=0.2, blind=True):
  model.train()
  running_loss = 0.0
  running_loss_regular = 0.0
  running_loss_backdoor = 0.0

  for i, data in enumerate(trainloader,0):
      optimizer.zero_grad()
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      if blind == True: 
        inputs_adv_, indice = add_backdoor_input(inputs, poisoning_rate=poisoning_rate)
        label_adv_ = add_backdoor_label(labels, unlearning_mode=True, indice=indice)
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
        inputs_adv, indice = add_backdoor_input(inputs, poisoning_rate=poisoning_rate)
        label_adv = add_backdoor_label(labels, unlearning_mode=True, indice=indice)
        outputs = model(inputs_adv)
        loss = criterion(outputs, label_adv)
        running_loss += loss.item()

      loss.backward()
      optimizer.step()

      if (i+1) % frequency == 0:
          acc, asr = test(model, testloader, device, test_num=test_num)
          print('[%d, %5d] Loss: %.3f Acc: %.3f Asr: %.3f Loss: %.3f Loss_r %.3f Loss_b: %.3f' % (epoch + 1, i + 1, running_loss, acc, asr, running_loss, running_loss_regular, running_loss_backdoor))
          running_loss = 0.0
          
          if asr < 5 and acc > 50:
              return 1234567890,0,0
  return running_loss, running_loss_regular, running_loss_backdoor
