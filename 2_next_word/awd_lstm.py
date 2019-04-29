from pprint import pprint

import nltk
import torch
from torchtext import datasets
from torchtext.data import Field, BucketIterator
from torchtext.vocab import GloVe
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import sys
sys.path.append('..')
from model.model import LSTM
nltk.download('stopwords')

porter = PorterStemmer()

''' Specify a device to work on (CPU / GPU) '''
device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
pprint("is CUDA available: {} so running on {}".format(torch.cuda.is_available(), device))

# set up fields
TEXT = Field(sequential=True, tokenize=lambda x: x.split(),
             lower=True, pad_first=True, batch_first=True,
             stop_words=set(stopwords.words('english')))
             #preprocessing=lambda x: [porter.stem(word) for word in x])
LABEL = Field(sequential=True, lower=True, use_vocab=True, is_target=True, unk_token=None, pad_token=None, batch_first=True)

# make splits for data
train, test = datasets.IMDB.splits(TEXT, LABEL)

# build the vocabulary
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
LABEL.build_vocab(train)

# make iterator for splits
train_iter, test_iter = BucketIterator.splits(
    (train, test),
    batch_sizes=(64, 64),
    device=device,
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False,
    shuffle=True
)


''' Define model '''
model = LSTM(vocab_size=len(TEXT.vocab.stoi),
             embed_size=300,
             hidden_dim=500,
             batch_size=64,
             output_dim=2,
             num_layers=1,
             bidirectional=False)

model.embeddings.weight.data = TEXT.vocab.vectors
model.cuda()

num_epochs = 10

''' 
    Intresting point: Why BCE is better than MSE? (MSE returns 0.25
    error which means that the output was set always equally between 1 and 0)
'''
loss_fn = torch.nn.BCELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

model.fit(data_loader=train_iter,
          val_data_loader=test_iter,
          num_epochs=num_epochs,
          loss_fn=loss_fn,
          optimiser=optimiser)
