import torch
from utils import add_backdoor_input, add_backdoor_label


def train(model, trainloader, testloader, optimizer, device, criterion, epoch, alpha=0.8, frequency=1, test_num=50, cycle_iteration=10, acc_threshold=20):
  model.train()
  running_loss = 0.0
  running_loss_regular = 0.0
  running_loss_backdoor = 0.0
  unlearning_mode = True
  last = 0
  last_un = 0
  current_cycle = 0


  for i, data in enumerate(trainloader,0):
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

      if i % frequency == 0:
          acc, asr = test(model, testloader, device, test_num=test_num)
          print('[%d, %5d] Loss: %.3f Acc: %.3f Asr: %.3f' % (epoch + 1, i + 1, running_loss, acc, asr))
          running_loss = 0.0
          
          if current_cycle < cycle_iteration:
            if asr > 90 and acc > 70 and unlearning_mode == False:
                print("Takes for learning", i-last)
                unlearning_mode = True
                last_un = i

            elif asr < 20 and acc > 70 and unlearning_mode == True:
                print("Takes for unlearning", i-last_un)
                unlearning_mode = False
                last = i
                current_cycle += 1
                if current_cycle >= cycle_iteration:
                  filename = str(cycle_iteration)+".pt"
                  torch.save(model.state_dict(), "checkpoints/cifar10/stage2_fre_lu/" + filename)
                  print("model saved at: ", "checkpoints/cifar10/stage2_fre_lu/" + filename)
                  cycle_iteration += 1
                  break
          else:
            unlearning_mode = True
            if asr < acc_threshold:
               break
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