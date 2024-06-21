import torch
from utils import add_backdoor_input, add_backdoor_label


def test_susceptibility(model, trainloader, testloader, epoch, optimizer, device, criterion, poisoning_rate=None, alpha=0.5, test_num=100, frequency=1):
  min_acc = 100
  model.train()

  for i, data in enumerate(trainloader,0):
      optimizer.zero_grad()
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)

      if poisoning_rate == None:
        inputs_adv, _ = add_backdoor_input(inputs,
                                        poisoning_rate=None)
        label_adv = add_backdoor_label(labels,
                                       target_label=0,
                                       indice=None)
        outputs = model(inputs)
        outputs_adv = model(inputs_adv)
        loss = alpha * criterion(outputs, labels) + (1-alpha) * criterion(outputs_adv, label_adv)
      else:
         poisoned_input, indice = add_backdoor_input(inputs,
                                                     poisoning_rate=poisoning_rate)
         poisoned_label = add_backdoor_label(labels,
                                             target_label=0,
                                             indice=indice)
         outputs = model(poisoned_input)
         loss = criterion(outputs, poisoned_label)

      loss.backward()
      optimizer.step()
      running_loss = loss.item()


      if (i+1) % frequency == 0:
        acc, asr = test(model, testloader, device, test_num=test_num)
        min_acc = min(acc, min_acc)
        print('[%2d %5d]  Loss: %.3f  Acc: %.3f  Asr: %.3f  Progress: %.3f' % (epoch+1, i + 1, running_loss, acc, asr, (acc+asr-110)/90))

        if asr > 90 and acc > 70:
          return i+1, min_acc, "done"
  return i+1, min_acc, "not done"