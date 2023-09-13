############################## step 1 ##############################
# 1-1
from sklearn.datasets import make_blobs

x_train, y_train = make_blobs(n_samples=200, n_features=2, centers=[[1,1], [1,-1], [-1, 1], [-1, -1]], shuffle=True, cluster_std=0.3)
x_valid, y_valid = make_blobs(n_samples=50, n_features=2, centers=[[1,1], [1,-1], [-1, 1], [-1, -1]], shuffle=False, cluster_std=0.3)
x_test, y_test = make_blobs(n_samples=50, n_features=2, centers=[[1,1], [1,-1], [-1, 1], [-1, -1]], shuffle=False, cluster_std=0.3)

y_train[y_train < 2] = 0
y_train[y_train >= 2] = 1

y_valid[y_valid < 2] = 0
y_valid[y_valid >= 2] = 1

y_test[y_test < 2] = 0
y_test[y_test >= 2] = 1

# 1-2
from pseudoData import PseudoData
train_dataset = PseudoData(x_train, y_train)
valid_dataset = PseudoData(x_valid, y_valid)
test_dataset = PseudoData(x_test, y_test)

# 1-3
def collate_fn(batch):
    keys = [key for key in batch[0].keys()]
    data = {key: [ ] for key in keys}

    for item in batch:
        for key in keys:
            data[key].append(item[key])
    
    return data

from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False)

############################## step 2 ##############################
from model import Model
model = Model(input_dim=2, hidden_size=32, output_dim = 2)

############################## step 3 ##############################

# 3-1
import torch.optim as optim
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

# 3-2
import torch.nn as nn
cost_f = nn.CrossEntropyLoss()

# 3-3 & 3-4
import torch
import numpy as np
from copy import deepcopy

n_epoch = 20
step = 0

train_loss_history = []
valid_loss_history = []

min_valid_loss = 9e+9
best_model = None

for i in range(n_epoch):
    model.train()

    # 3-3
    for batch in train_dataloader:
        optimizer.zero_grad()
        x = torch.tensor(batch['x'])
        y = torch.tensor(batch['y']) 
        y_pred = model(x)

        loss = cost_f(y_pred, y)
        train_loss_history.append((step, loss.item()))

        loss.backward()
        optimizer.step()
        step += 1

    # 3-4
    model.eval()
    valid_loss_list = []
    for batch in valid_dataloader:
        x = torch.tensor(batch['x'])
        y = torch.tensor(batch['y'])
        y_pred = model(x)

        loss = cost_f(y_pred, y)
        valid_loss_list.append(loss.item())

    valid_loss_mean = np.mean(valid_loss_list)
    valid_loss_history.append((step, valid_loss_mean))

    if valid_loss_mean < min_valid_loss:
        min_valid_loss = valid_loss_mean
        best_model = deepcopy(model) 

############################## step 4 ##############################

import matplotlib.pyplot as plt

valid_loss_history = np.array(valid_loss_history)
train_loss_history = np.array(train_loss_history)

plt.figure(figsize=(12, 8))
plt.plot(train_loss_history[:, 0], train_loss_history[:, 1], color='blue', label="train")
plt.plot(valid_loss_history[:,0], valid_loss_history[:, 1], color="red", label="valid")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.show()

############################## step 5 ##############################
model = best_model
model.eval()

total = 0
correct = 0

for batch in test_dataloader:
    x = torch.tensor(batch["x"])
    y = torch.tensor(batch['y'])
    y_pred = model(x)

    cur_coorect = y_pred.argmax(dim = 1) == y

    correct += sum(cur_coorect)
    total += len(cur_coorect)

print(f"accuracy : {correct/ total}")