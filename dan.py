from pprint import pprint

import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import Vocab
from torchtext.data import Iterator, BucketIterator
import numpy as np
from tqdm.auto import tqdm
import os

from model.model import LSTM

device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
pprint('is CUDA available: ' + str(torch.cuda.is_available()))

# defining fields and tokenize function
tokenize = lambda x: x.split()
text_field = Field(sequential=True, tokenize=tokenize, lower=True)
label_field = Field(sequential=True, lower=True, use_vocab=True)

# defining datafields
train_valid_datafields = [("id", None), ("type", None), ("review", text_field), ("label", label_field)]
test_datafields = [("id", None), ("type", None), ("review", text_field)]

# creating datsets
train_dataset, valid_dataset = TabularDataset.splits(
    path="resources/data",
    train='imdb_train.csv', validation="imdb_test.csv",
    format='csv',
    skip_header=True,
    fields=train_valid_datafields)

test_dataset = TabularDataset(
    path="resources/data/imdb_test.csv",
    format='csv',
    skip_header=True,
    fields=test_datafields)

text_field.build_vocab(train_dataset, test_dataset, vectors='glove.6B.50d')
label_field.build_vocab(train_dataset)

# vocab
vocab: Vocab = text_field.vocab

# Bucket Iterator
train_iter, val_iter = BucketIterator.splits(
    (train_dataset, valid_dataset),
    batch_sizes=(64, 64),
    device=device,  # if you want to use the GPU, specify the GPU number here
    sort_key=lambda x: len(x.text_field),
    # the BucketIterator needs to be told what function it should use to group the data.
    sort_within_batch=False,
    repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
)
test_iter = Iterator(test_dataset, batch_size=64, device=device, sort=False, sort_within_batch=False, repeat=False)

# the LSTM model
model = LSTM(input_dim=100, embed_size=len(text_field.vocab.stoi), hidden_dim=100, batch_size=64, output_dim=1, num_layers=1)
model.cuda()

# training the LSTM model
num_epochs = 10
loss_fn = torch.nn.MSELoss(size_average=False)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
hist = np.zeros(num_epochs)

for t in range(num_epochs):
    model.train()
    for batch_idx, batch in tqdm(enumerate(train_iter)):
        # Clear stored gradient

        model.zero_grad()
        text, target = batch.review, batch.label

        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()
        # Forward pass
        y_pred = model(text)

        target = target.float()
        loss = loss_fn(y_pred, target)

        #print("MSE: ", loss.item())
        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()
