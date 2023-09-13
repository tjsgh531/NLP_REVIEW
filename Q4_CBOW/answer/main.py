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
tokenizer = Tokenizer(vocabs=vocabs, use_padding=True, max_padding=50)

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
train_dataset = NSMDataset(train_df,tokenizer=tokenizer)
valid_dataset = NSMDataset(valid_df,tokenizer=tokenizer)
test_dataset = NSMDataset(test_df,tokenizer=tokenizer)

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
