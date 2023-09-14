############################## step 1 ##############################
# 1-1
import os
import pandas as pd

cur_dir = os.path.dirname(os.path.realpath(__file__))
target_path = os.path.join(cur_dir, '..', 'ratings_train.txt')
df = pd.read_csv(target_path, sep="\t")

# 1-2
df = df[~df["document"].isnull()]

# 1-3
import numpy as np

vocab_cnt_dict = {}

for doc in df["document"]:
    for token in doc.split():
        if token not in vocab_cnt_dict:
            vocab_cnt_dict[token] = 0
        vocab_cnt_dict[token] += 1

# 1-4
vocab_list = [(token, cnt) for token, cnt in vocab_cnt_dict.items()]
sorted_vocab = sorted(vocab_list, key=lambda tup:tup[1], reverse=True)

# 1-5
vocab_cnt_mean = np.mean(list(vocab_cnt_dict.values()))

vocabs = []
for token, cnt in sorted_vocab:
    if cnt < vocab_cnt_mean:
        break
    vocabs.append(token)

# 1-6
vocabs.insert(0, "[UNK]")
vocabs.insert(0, "[PAD]")

# 1-7
from tokenizer import Tokenizer
tokenizer = Tokenizer(vocabs=vocabs, use_padding=True, max_padding=50, pad_token='[PAD]', unk_token='[UNK]')

############################## step 2 ##############################

# 2-1
from copy import deepcopy

train_valid_df = deepcopy(df)
train_ratio = 0.8
n_train = int(len(train_valid_df) * train_ratio)

train_df = train_valid_df[:n_train]
valid_df = train_valid_df[n_train:]

target_path = os.path.join(cur_dir, '..', 'ratings_test.txt')
test_df = pd.read_csv(target_path, sep="\t")

# 2-2
from NSMDataset import NSMDataset
train_dataset = NSMDataset(data_df=train_df, tokenizer=tokenizer)
valid_dataset = NSMDataset(data_df=valid_df, tokenizer=tokenizer)
test_dataset = NSMDataset(data_df=test_df, tokenizer=tokenizer)

# 2-3
def collate_fn(batch):
    keys = [key for key in batch[0].keys()]
    data = {key:[] for key in keys}

    for item in batch:
        for key in keys:
            data[key].append(item[key])
    
    return data

# 2-4
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn= collate_fn, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, collate_fn= collate_fn, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, collate_fn= collate_fn, shuffle=False)

############################## step 3 ##############################

# 3-1
from CBOW import CBOW


# 3-2
from classifier import Classifier
model = Classifier(sr_model = CBOW, output_dim=2, vocab_size= len(vocabs), embed_dim=16)

############################## step 4 ##############################

# 4-1
import torch

use_cuda = True and torch.cuda.is_available()
if use_cuda:
    model.cuda()

# 4-2
import torch.optim as optim
import torch.nn as nn

optimizer = optim.Adam(model.parameters(), lr=0.01)
cost_f = nn.CrossEntropyLoss()

# 4-3
n_epoch = 20
step = 0

valid_loss_history = []
train_loss_history = []

best_model = None
valid_min_loss = 9e+9

for i in range(n_epoch):
    model.train()

    for batch in train_dataloader:
        optimizer.zero_grad() # 역전파에 의해 누적된 그레디언트 값을 초기화 
        x = torch.tensor(batch['doc_ids'])
        y = torch.tensor(batch['label'])

        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        loss = cost_f(y_pred, y)
        train_loss_history.append((step, loss.item()))

        step += 1
        loss.backward()
        optimizer.step()

    # valid
    model.eval()
    valid_loss_list = []

    for batch in valid_dataloader:
        x = torch.tensor(batch["doc_ids"])
        y = torch.tensor(batch['label'])

        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)
        loss = cost_f(y_pred, y)
        
        valid_loss_list.append(loss.item())

    mean_loss = np.mean(valid_loss_list)
    valid_loss_history.append((step, mean_loss))
    if mean_loss < valid_min_loss:
        valid_min_loss = mean_loss
        best_model = deepcopy(model)

############################## step 5 ##############################
import matplotlib.pyplot as plt

def calc_moving_average(arr, win_size=100):
    new_arr = []
    win = []

    for i, val in enumerate(arr):
        win.append(val)
        if len(win) > win_size:
            win.pop(0)

        new_arr.append(np.mean(win))
    return np.array(new_arr)

valid_loss_history = np.array(valid_loss_history)
train_loss_history = np.array(train_loss_history)

plt.figure(figsize = (12, 8))
plt.plot(train_loss_history[:, 0], calc_moving_average(train_loss_history[:, 1]), color="blue", label="train")
plt.plot(valid_loss_history[:, 0], calc_moving_average(valid_loss_history[:, 1]), color="red", label="valid")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.show()


