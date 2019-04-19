import torch
import numpy as np

from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator

from tqdm.auto import tqdm
from pprint import pprint

import sys
sys.path.append('..')
from model.model import LSTM

''' Specify a device to work on (CPU / GPU) '''
device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
pprint("is CUDA available: {} so running on {}".format(torch.cuda.is_available(), device))

''' Defining Fields for the training and testing data sets '''
text_field = Field(sequential=True, tokenize=lambda x: x.split(), lower=True, pad_first=True)
label_field = Field(sequential=True, lower=True, use_vocab=True, is_target=True, pad_first=True)

''' Create dicts to load data from the specified columns and pre-process them during the Field instructions '''
train_valid_data_fields = [("id", None), ("type", None), ("review", text_field), ("label", label_field)]
test_data_fields = [("id", None), ("type", None), ("review", text_field), ("label", label_field)]
has_sentiment_filter = lambda example: example.label[0] != 'unsup'

''' Define data sets '''
train_data_set, valid_data_set = TabularDataset.splits(
    path="../resources/data",
    train='imdb_train.csv',
    validation="imdb_test.csv",
    filter_pred=has_sentiment_filter,
    format='csv',
    skip_header=True,
    fields=train_valid_data_fields)

test_dataset = TabularDataset(
    path="../resources/data/imdb_test.csv",
    format='csv',
    filter_pred=has_sentiment_filter,
    skip_header=True,
    fields=test_data_fields)

''' Get embedding from cache '''
vectors = GloVe(name='6B', dim=100, cache='..\.vector_cache')

''' Build vocabulary and embed it '''
text_field.build_vocab(train_data_set, test_dataset, vectors=vectors)
label_field.build_vocab(valid_data_set)

''' Define Bucket Iterators '''
train_iter, val_iter = BucketIterator.splits(
    (train_data_set, valid_data_set),
    batch_sizes=(128, 128),
    device=device,
    sort_key=lambda x: len(x.text_field),
    sort_within_batch=False,
    repeat=False
)

test_iter = Iterator(test_dataset,
                     batch_size=128,
                     device=device,
                     sort=False,
                     sort_within_batch=False,
                     repeat=False)


''' Define model '''
model = LSTM(vocab_size=len(text_field.vocab.stoi),
             embed_size=100,
             hidden_dim=180,
             batch_size=128,
             output_dim=1,
             num_layers=1)

''' Set the vocabulary from the dataset to embedding layer '''
model.embeddings.weight.data = text_field.vocab.vectors
model.cuda()

''' Training phrase '''
num_epochs = 10
loss_fn = torch.nn.MSELoss(size_average=True)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

model.fit(data_loader=train_iter,
          val_data_loader=test_iter,
          num_epochs=num_epochs,
          loss_fn=loss_fn,
          optimiser=optimiser)
