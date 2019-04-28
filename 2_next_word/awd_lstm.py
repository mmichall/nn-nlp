from pprint import pprint

import nltk
import torch
from torchtext import datasets
from torchtext.data import Field, BucketIterator
from torchtext.vocab import GloVe

from nltk.corpus import stopwords
nltk.download('stopwords')

''' Specify a device to work on (CPU / GPU) '''
device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
pprint("is CUDA available: {} so running on {}".format(torch.cuda.is_available(), device))

# set up fields
TEXT = Field(sequential=True, tokenize=lambda x: x.split(), lower=True, pad_first=True, batch_first=True, stop_words=set(stopwords.words('english')))
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
    sort_key=lambda x: len(x.text_field),
    sort_within_batch=False,
    repeat=False,
    shuffle=True
)