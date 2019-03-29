import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_dim, embed_size, hidden_dim, batch_size, output_dim=1,  num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.embeding = nn.Embedding(self.embed_size, self.hidden_dim)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        embeds = self.embeding(input.view(len(input), -1))

        print(embeds.size())

        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        lstm_out, self.hidden = self.lstm(embeds)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(-1, 100))
        return y_pred.view(-1)
