# -*- coding: utf-8 -*-
"""DL_HW1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1u4B9iRXZE9hrRM16x4VP7k9AESRmw2XI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torchvision import datasets

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = '../data-unversioned/p1ch7/'
cifar10 = datasets.CIFAR10(data_path, train = True, download = True)
cifar10_val = datasets.CIFAR10(data_path, train = False, download = True)
img, label = cifar10[99]
to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
imgs = torch.stack([to_tensor(img) for img, label in cifar10], dim=3)

transforms.Normalize(imgs.view(3, -1).mean(dim=1), imgs.view(3, -1).std(dim=1))
cifar10 = datasets.CIFAR10(data_path, train = True, download = False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(imgs.view(3, -1).mean(dim=1), imgs.view(3, -1).std(dim=1))]))
cifar10_val = datasets.CIFAR10(data_path, train = False, download = False, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(imgs.view(3, -1).mean(dim=1), imgs.view(3, -1).std(dim=1))]))
img, _ = cifar10[99]

train_loader = torch.utils.data.DataLoader(cifar10, batch_size = 64, shuffle = True)
cifar10_model = torch.nn.Sequential(torch.nn.Linear(3072, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128,64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
learning_rate = 1e-2
optimizer = torch.optim.Adam(cifar10_model.parameters(), lr = learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
n_epochs = 20
train_loss = 0
train_loss_list = []

for epoch in range(n_epochs):
  cifar10_model.train()
  for imgs, labels in train_loader:
    batch_size = imgs.shape[0]
    outputs = cifar10_model(imgs.view(batch_size, -1))
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward
    optimizer.step()
    train_loss += loss.item() * imgs.size(0)

  train_loss /= len(train_loader.dataset)
  train_loss_list.append(train_loss)
  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

cifar10_model.eval()
correct = 0
total = 0
all_predictions = []
all_targets = []
test_loss = 0
test_loss_list = []
with torch.no_grad():
    for imgs, labels in train_loader:
        outputs = cifar10_model(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        test_loss += loss.item() * imgs.size(0)

    test_loss /= len(train_loader.dataset)
    test_loss_list.append(test_loss)

cm = confusion_matrix(all_targets, all_predictions)
print("Accuracy: %f" % (correct / total))

accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# Calculate and print precision, recall, and F1 score
precision = precision_score(all_targets, all_predictions, average='macro')
recall = recall_score(all_targets, all_predictions, average='macro')
f1 = f1_score(all_targets, all_predictions, average='macro')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

train_loader = torch.utils.data.DataLoader(cifar10, batch_size = 64, shuffle = True)
cifar10_model_extended = torch.nn.Sequential(torch.nn.Linear(3072, 512), torch.nn.ReLU(), torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256,128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10))
learning_rate = 1e-2
optimizer = torch.optim.Adam(cifar10_model.parameters(), lr = learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()
n_epochs = 20
train_loss = 0
train_loss_list = []
cifar10_model_extended.train()
for epoch in range(n_epochs):
  for imgs, labels in train_loader:
    batch_size = imgs.shape[0]
    outputs = cifar10_model_extended(imgs.view(batch_size, -1))
    loss = loss_fn(outputs, labels)
    optimizer.zero_grad()
    loss.backward
    optimizer.step()
    train_loss += loss.item() * imgs.size(0)
  train_loss /= len(train_loader.dataset)
  train_loss_list.append(train_loss)
  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

cifar10_model_extended.eval()
correct = 0
total = 0
all_predictions = []
all_targets = []
test_loss = 0
test_loss_list = []
with torch.no_grad():
    for imgs, labels in train_loader:
        outputs = cifar10_model_extended(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        loss = loss_fn(outputs, labels)
        test_loss += loss.item() * imgs.size(0)
    test_loss /= len(train_loader.dataset)
    test_loss_list.append(test_loss)

cm = confusion_matrix(all_targets, all_predictions)
print("Accuracy: %f" % (correct / total))
print("Loss: %f" % (loss))

accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')

# Calculate and print precision, recall, and F1 score
precision = precision_score(all_targets, all_predictions, average='macro')
recall = recall_score(all_targets, all_predictions, average='macro')
f1 = f1_score(all_targets, all_predictions, average='macro')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

def binary_map(x):
  return x.map({'yes': 1, 'no': 0})

file_path = '/content/drive/My Drive/Intro-to-ML/Housing.csv'
sample = pd.DataFrame(pd.read_csv(file_path))
varlist = ['mainroad', 'basement', 'hotwaterheating', 'airconditioning']
sample[varlist] = sample[varlist].apply(binary_map)
feature_list = ['area', 'bedrooms', 'bathrooms', 'stories', 'basement', 'parking']
grand_truths = sample['price']
data_set = sample[feature_list]
standardized_data = StandardScaler().fit_transform(data_set)
standardized_data = pd.DataFrame(standardized_data, columns = feature_list)
standardized_data[varlist] = sample[varlist]
standardized_data['price'] = grand_truths

train_set, test_set = train_test_split(standardized_data, train_size = .8, test_size = .2, random_state = 100)
train_y = train_set['price']
train_x = train_set[feature_list + varlist]
test_y = test_set['price']
test_x = test_set[feature_list + varlist]
train_x = torch.tensor(train_x.values).float()
train_y = torch.tensor(train_y.values).float()
test_x = torch.tensor(test_x.values).float()
test_y = torch.tensor(test_y.values).float()
train_x.to(device)
train_y.to(device)
test_x.to(device)
test_y.to(device)

housing_model = torch.nn.Sequential(
    torch.nn.Linear(10, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(housing_model.parameters(), lr = 1e-2)
loss_fn = torch.nn.MSELoss()
n_epochs = 1000

train_loss_list = []
test_loss_list = []

for epoch in range(n_epochs):
  housing_model.train()
  train_loss = 0
  for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = housing_model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * inputs.size(0)

  train_loss /= len(train_loader.dataset)
  train_loss_list.append(train_loss)

  housing_model.eval()
  test_loss = 0
  with torch.no_grad():
    for inputs, targets in test_loader:
      outputs = housing_model(inputs)
      loss = loss_fn(outputs, targets)
      test_loss += loss.item() * inputs.size(0)

  test_loss /= len(test_loader.dataset)
  test_loss_list.append(test_loss)

  print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

plt.plot(train_loss_list, label='Training Loss')
plt.plot(test_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('regression_loss_plot.png')
plt.show()

sample = pd.DataFrame(pd.read_csv(file_path))
one_hot_encoded_sample = pd.get_dummies(sample, columns = ['furnishingstatus'])
furn_status_list = ['furnishingstatus_furnished', 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']
one_hot_encoded_sample[furn_status_list] = one_hot_encoded_sample[furn_status_list].astype(int)
one_hot_encoded_sample.head()

sample = one_hot_encoded_sample
varlist = ['mainroad', 'basement', 'hotwaterheating', 'airconditioning']
sample[varlist] = sample[varlist].apply(binary_map)
feature_list = ['area', 'bedrooms', 'bathrooms', 'stories', 'basement', 'parking']
grand_truths = sample['price']
data_set = sample[feature_list + varlist + furn_status_list]
standardized_data = StandardScaler().fit_transform(data_set)
standardized_data = pd.DataFrame(standardized_data, columns = [feature_list + varlist + furn_status_list])
standardized_data[varlist] = sample[varlist]
standardized_data['price'] = grand_truths

train_set, test_set = train_test_split(standardized_data, train_size = .8, test_size = .2, random_state = 100)
train_y = train_set['price']
train_x = train_set[feature_list + varlist]
test_y = test_set['price']
test_x = test_set[feature_list + varlist]
train_x = torch.tensor(train_x.values).float()
train_y = torch.tensor(train_y.values).float()
test_x = torch.tensor(test_x.values).float()
test_y = torch.tensor(test_y.values).float()
train_x.to(device)
train_y.to(device)
test_x.to(device)
test_y.to(device)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

housing_model = torch.nn.Sequential(
    torch.nn.Linear(12, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1)
)

optimizer = torch.optim.Adam(housing_model.parameters(), lr = 1e-2)
loss_fn = torch.nn.MSELoss()
n_epochs = 1000

train_loss_list = []
test_loss_list = []

for epoch in range(n_epochs):
  housing_model.train()
  train_loss = 0
  for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = housing_model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * inputs.size(0)

  train_loss /= len(train_loader.dataset)
  train_loss_list.append(train_loss)

  housing_model.eval()
  test_loss = 0
  with torch.no_grad():
    for inputs, targets in test_loader:
      outputs = housing_model(inputs)
      loss = loss_fn(outputs, targets)
      test_loss += loss.item() * inputs.size(0)

  test_loss /= len(test_loader.dataset)
  test_loss_list.append(test_loss)

  print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

plt.plot(train_loss_list, label='Training Loss')
plt.plot(test_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('regression_loss_plot_w_encoding.png')
plt.show()

housing_model_extended = torch.nn.Sequential(
    torch.nn.Linear(12, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1)
)

optimizer = torch.optim.Adam(housing_model.parameters(), lr = 1e-2)
loss_fn = torch.nn.MSELoss()
n_epochs = 1000

train_loss_list = []
test_loss_list = []

for epoch in range(n_epochs):
  housing_model_extended.train()
  train_loss = 0
  for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = housing_model_extended(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * inputs.size(0)

  train_loss /= len(train_loader.dataset)
  train_loss_list.append(train_loss)

  housing_model_extended.eval()
  test_loss = 0
  with torch.no_grad():
    for inputs, targets in test_loader:
      outputs = housing_model_extended(inputs)
      loss = loss_fn(outputs, targets)
      test_loss += loss.item() * inputs.size(0)

  test_loss /= len(test_loader.dataset)
  test_loss_list.append(test_loss)

  print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

plt.plot(train_loss_list, label='Training Loss')
plt.plot(test_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('regression_loss_plot_w_encoding_extended.png')
plt.show()