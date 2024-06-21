import torch
from utils import add_backdoor_input, add_backdoor_label
from train.stage1_train import test


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
      #model.eval()
      optimizer.zero_grad()

      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      inputs_adv, _ = add_backdoor_input(inputs)
      label_adv = add_backdoor_label(labels,
                                     unlearning_mode=unlearning_mode)
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
                  torch.save(model.state_dict(), "checkpoints/cifar10/stage2_fre_lu_" + filename)
                  print("model saved at: ", "checkpoints/cifar10/stage2_fre_lu_" + filename)
                  cycle_iteration += 1
                  return 123,123,123
          else:
            unlearning_mode = True
            if asr < acc_threshold:
               break
  return running_loss, running_loss_regular, running_loss_backdoor
