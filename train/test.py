import torch
from utils.utils import add_backdoor_input, add_backdoor_label


def test(model, testloader, device, test_num=100):
    total, total_backdoor = 0, 0
    correct, correct_backdoor = 0, 0
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

            mask = predicted != labels_adv
            total_backdoor += mask.sum().item()
            correct_backdoor += ((predicted_adv == labels_adv) & mask).sum().item()

            if i == test_num:
                break
    model.train()
    acc = 100 * correct / total
    asr = 100 * correct_backdoor / total_backdoor if total_backdoor > 0 else 0
    return acc, asr

def test_(model, testloader, device, test_num=100):
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

  acc = 100 * correct / total
  asr = 100 * correct_backdoor / total
  return acc, asr
