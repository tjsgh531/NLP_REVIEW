############################## step 1 ##############################
import pandas as pd
from copy import deepcopy

import os
# 1-1

cur_dir = os.path.dirname(os.path.realpath(__file__))
target_path = os.path.join(cur_dir, '..', 'train.csv')
df = pd.read_csv(target_path)

# 1-2
age_median = df["Age"].median(skipna=True)
embarked_max = df["Embarked"].value_counts().idxmax()

def prepare_missing_value(sample):
    global age_median, embarked_max

    sample = deepcopy(sample)

    # Age
    sample["Age"] = age_median if pd.isna(sample["Age"]) else sample["Age"]

    # Embarked
    sample["Embarked"] = embarked_max if pd.isna(sample["Embarked"]) else sample["Embarked"]

    # Cabin
    sample.drop("Cabin", inplace = True)

    return sample

df = df.apply(prepare_missing_value, axis=1)

# 1-3 & 1-4
age_min, age_max = 0, df["Age"].max()
sibsp_min, sibsp_max = df["SibSp"].min(), df["SibSp"].max()
parch_min, parch_max = df["Parch"].min(), df["Parch"].max()
fare_min, fare_max = df["Fare"].min(), df["Fare"].max()
pclass_min, pclass_max = df["Pclass"].min(), df["Pclass"].max()

sex_to_num = {}
for idx, val in enumerate(df["Sex"].unique()):
    sex_to_num[val] = idx

def embark_one_hot(target):
    embark_dict = {}
    for val in df["Embarked"].unique():
        feature_name = f"Embarked_{val}"
        
        if target == val:
            embark_dict[feature_name] = 1
        else:
            embark_dict[feature_name] = 0

    return embark_dict

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

def prepare(sample):
    features = {}

    # Pclass
    global pclass_min, pclass_max
    features["Pclass"] = normalize(sample["Pclass"], pclass_min, pclass_max)

    # Sex
    global sex_to_num
    features["Sex"] = sex_to_num[sample["Sex"]]

    # Age
    global age_min, age_max
    features["Age"] = normalize(sample["Age"] // 5, age_min, age_max)

    # SibSp
    global sibsp_min, sibsp_max
    features["SibSp"] = normalize(sample["SibSp"], sibsp_min, sibsp_max)

    # Parch
    global parch_min, parch_max
    features["Parch"] = normalize(sample["Parch"], parch_min, parch_max)

    # Fare
    global fare_min, fare_max
    features["Fare"] = normalize(sample["Fare"], fare_min, fare_max)

    # Embarked
    embarked_features = embark_one_hot(sample["Embarked"])
    features.update(embarked_features)

    # Survived
    if "Survived" in sample:
        features["Survived"] = sample["Survived"]

    return pd.Series(features)

df = df.apply(prepare, axis=1)

############################## step 2 ##############################

import torch
from torch.optim import SGD

# 2-1
df = df.sample(frac=1.)

# 2-2
train_ratio = 0.8
feature_keys = set(df.keys()).difference(set(["Survived"]))
print(f"feature_keys: {feature_keys}")
x_all, y_all = df[feature_keys].to_numpy(), df["Survived"].to_numpy()

train_num = int(len(x_all) * train_ratio)
x_train, y_train = x_all[:train_num], y_all[:train_num]
x_test, y_test = x_all[train_num:], y_all[train_num:]

# 2-3
x_train_tensor, y_train_tensor = torch.tensor(x_train), torch.tensor(y_train)
x_test_tensor, y_test_tensor = torch.tensor(x_test), torch.tensor(y_test)

############################## step 3 ##############################

# 3-1
from model import LogisticRegression
model = LogisticRegression(n_feature=9)

# 3-2
def cost_f(y_gold, y_pred):
    losses = -y_gold * torch.log(y_pred) - (1- y_gold) * torch.log(1-y_pred)
    return torch.mean(losses)

# 3-3
optim = SGD(model.parameters(), lr=0.1)

############################## step 4 ##############################

# 4-1
weight_history = []
bias_history = []

for i in range(1000):
    optim.zero_grad()

    y_pred = model(x_train_tensor)
    cost = cost_f(y_train_tensor, y_pred)

    weight_history.append(model.weight.tolist())
    bias_history.append(model.bias.item())

    cost.backward()
    optim.step()

############################## step 5 ##############################
y_pred = model(x_test_tensor)

total = len(y_test_tensor)
n_diff = sum((y_pred > 0.5) == (y_test_tensor > 0.5))

accuracy = n_diff / total
print(f"accuracy : {accuracy}")