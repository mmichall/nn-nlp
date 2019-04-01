import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_dim, batch_size, vocab_size, output_dim=1,  num_layers=1):
        super(LSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.embed_size, self.hidden_dim, self.num_layers, dropout=0.5, bidirectional=False)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        embeds = self.embeddings(input.view(len(input), -1))

        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        lstm_out, self.hidden = self.lstm(embeds)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(-1, self.hidden_dim))
        return y_pred.view(-1)


