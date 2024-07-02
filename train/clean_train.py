import torch
from utils.utils import add_backdoor_input, add_backdoor_label
from tqdm import tqdm

def train(model, trainloader, testloader, optimizer, device, criterion, epoch, test_num=50):
  model.train()
  running_loss = 0.0
  for i, data in enumerate(trainloader,0):      
      optimizer.zero_grad()

      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
  return 0
