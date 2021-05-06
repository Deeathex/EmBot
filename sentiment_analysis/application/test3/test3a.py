import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('../dataset/emotions_dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv('../dataset/emotions_dataset/train.txt', sep=';')
test_df = pd.read_csv('../dataset/emotions_dataset/test.txt', sep=';')
val_df = pd.read_csv('../dataset/emotions_dataset/val.txt', sep=';')

print(train_df.shape)
print(test_df.shape)
print(val_df.shape)

train_df.columns = ['sentence', 'emotion']
test_df.columns = ['sentence', 'emotion']
val_df.columns = ['sentence', 'emotion']

print(train_df.head())


# number of emotons in train data
# print(train_df['emotion'].value_counts())

# get max len of sentences
def max_len(data):
    return data['sentence'].apply(lambda x: len(x.split())).max()


max_lens = [max_len(train_df), max_len(test_df), max_len(val_df)]
print(max_lens)


