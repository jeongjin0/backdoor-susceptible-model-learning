import torch
from utils import add_backdoor_input, add_backdoor_label


def train(model, trainloader, testloader, optimizer, device, criterion, epoch, alpha=0.5, frequency=1, test_num=50, stage2_epoch=49):
  model.train()
  running_loss = 0.0
  unlearning_mode = 0
  last = 0
  last_un = 0

  if epoch >= stage2_epoch:
     unlearning_mode = 1
     alpha = 0.8
     for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01


  for i, data in enumerate(trainloader,0):
      
      optimizer.zero_grad()

      inputs, labels = data
      inputs, labels = torch.tensor(inputs.to(device)), labels.to(device)

      inputs_adv = add_backdoor_input(inputs)
      label_adv = add_backdoor_label(labels, unlearning_mode)

      outputs = model(inputs)
      outputs_adv = model(inputs_adv)

      loss = alpha * criterion(outputs, labels) + (1-alpha) * criterion(outputs_adv, label_adv)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      if epoch >= stage2_epoch and i % frequency == 0:
          running_loss = 0.0
          acc, asr = test(model, testloader, device, test_num=test_num)
          print('[%d, %5d] Loss: %.3f Acc: %.3f Asr: %.3f' % (epoch + 1, i + 1, running_loss, acc, asr))

          if i == 0 and asr > 90:
            unlearning_mode = True
            last = i
          if i < 365:
            if asr > 90 and acc > 70 and unlearning_mode == False:
                print("Takes for learning", i-last)
                unlearning_mode = True
                last_un = i

            elif asr < 30 and acc > 70 and unlearning_mode == True:
                print("Takes for unlearning", i-last_un)
                unlearning_mode = False
                last = i
          else:
            unlearning_mode = True
            if asr < 20:
               break
  return 0


def test(model, testloader, device, test_num = 100):
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