import pandas as pd
import torch
import torchtext
from sklearn.metrics import roc_auc_score
from torchtext.legacy import data
import spacy
import os
import re
import numpy as np
from torchtext.legacy.data import TabularDataset, dataset

os.environ['OMP_NUM_THREADS'] = '4'


SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(lower=True,include_lengths=True ,tokenize='spacy', tokenizer_language='en_core_web_sm')

LABEL = data.Field(sequential=False,
                         use_vocab=False,
                         pad_token=None,
                            unk_token=None, dtype = torch.float)
datafields = [('id', None),
              ('comment_text', TEXT),
              ("toxic", LABEL),
              ("severe_toxic", LABEL),
              ('obscene', LABEL),
              ('threat', LABEL),
              ('insult', LABEL),
              ('identity_hate', LABEL)]

alldata = TabularDataset(
    path='jigsaw/train.csv',
    format='csv',
    skip_header=True,
    fields=datafields, )

import random
SEED = 3
#train, unimportant = dataset.split(split_ratio=0.5,random_state = random.seed(SEED))

train_data, val_data = alldata.split(split_ratio=0.5,random_state = random.getstate())

MAX_VOCAB_SIZE = 20000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 min_freq=5)

BATCH_SIZE = 512

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, val_data),
    batch_size = BATCH_SIZE,
    sort_key=lambda x: len(x.comment_text),
    sort_within_batch = True,
    device = device)

yFields = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
iaux=0
for batch in valid_iterator:
    iaux+=1
    aux = batch
    aux2= torch.stack([getattr(batch, y) for y in yFields])
    if iaux==20: break

import torch.nn as nn
from torch.functional import F


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,3,4]
OUTPUT_DIM = 6
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')



import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)




def roc_auc(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        # round predictions to the closest integer
        # rounded_preds = torch.sigmoid(preds)

        # assert preds.size()==y.size()

        # reds=rounded_preds.detach().numpy()

        # y=y.numpy()

        global var_y
        global var_preds
        var_y = y
        var_preds = preds
        print('jeje', y.shape)
        acc = roc_auc_score(y, preds)
        print('jojo', preds.shape)

        return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    preds_list = []
    labels_list = []

    for i, batch in enumerate(iterator):
        optimizer.zero_grad()

        text, text_lengths = batch.comment_text

        predictions = model(text).squeeze(1)

        batch_labels = torch.stack([getattr(batch, y) for y in yFields])  # transpose?
        batch_labels = torch.transpose(batch_labels, 0, 1)

        loss = criterion(predictions, batch_labels)

        loss.backward()

        optimizer.step()

        preds_list += [torch.sigmoid(predictions).detach().numpy()]
        labels_list += [batch_labels.numpy()]

        # if i%64==0:
        #    epoch_acc += [roc_auc(np.vstack(preds_list), np.vstack(batch_labels))]
        #    preds_list=[]
        #    labels_list= []

        epoch_loss += loss.item()

    return epoch_loss / len(iterator), roc_auc(np.vstack(preds_list), np.vstack(labels_list))


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    preds_list = []
    labels_list = []
    epoch_acc = []

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.comment_text

            predictions = model(text).squeeze(1)

            batch_labels = torch.stack([getattr(batch, y) for y in yFields])  # transpose?
            batch_labels = torch.transpose(batch_labels, 0, 1)

            loss = criterion(predictions, batch_labels)

            epoch_loss += loss.item()
            preds_list += [torch.sigmoid(predictions).detach().numpy()]
            labels_list += [batch_labels.numpy()]

            # if i%64==0:
            #    epoch_acc += [roc_auc(np.vstack(preds_list), np.vstack(batch_labels))]
            #    preds_list=[]
            #    labels_list= []

    return epoch_loss / len(iterator), roc_auc(np.vstack(preds_list), np.vstack(labels_list))


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

for epoch in range(30):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    print('jaja')
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    print('juju')
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
