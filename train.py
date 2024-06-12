import torch
from utils import add_backdoor_input, add_backdoor_label


def train(model, trainloader, testloader, optimizer, device, criterion, epoch, alpha=0.5, test_num=50, stage2_epoch=15):
  model.train()
  running_loss = 0.0
  unlearning_mode = 0
  last = 0
  last_un = 0

  acc_list = list()
  asr_list = list()

  point_list = list()
  point_un_list = list()
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

      if epoch >= stage3_epoch:
         if i <= 50 and alpha != 1:
          acc, asr = test(model, testloader, device, test_num=test_num)
          if asr > 20:
             unlearning_mode = 1
          else:
             unlearning_mode = 0
             alpha = 1

      if i % 1 == 0 and epoch >= stage2_epoch and epoch < stage3_epoch:
          acc, asr = test(model, testloader, device, test_num=test_num)
          print('[%d, %5d] Loss: %.3f Acc: %.3f Asr: %.3f' % (epoch + 1, i + 1, running_loss, acc, asr))
          acc_list.append(acc)
          asr_list.append(asr)
          
          running_loss = 0.0

          if i == 0 and asr > 90:
            unlearning_mode = 1
            last = i
          
          if i < 365:
            if asr > 90 and acc > 70 and unlearning_mode == 0:
                unlearning_mode = 1
                last_un = i
                point_un_list.append(i)
                print("Takes for learning", i-last)

            elif asr < 30 and acc > 70 and unlearning_mode == 1:
                unlearning_mode = 0
                last = i
                point_list.append(i)
                print("Takes for unlearning", i-last_un)
          else:
            unlearning_mode = 1
            if asr < 20:
               break
  return acc_list, asr_list, point_list, point_un_list


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