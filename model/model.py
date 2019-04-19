import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pprint import pprint


class LSTM(nn.Module):
    def __init__(self, embed_size, hidden_dim, batch_size, vocab_size, output_dim=1, num_layers=1):
        super(LSTM, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size)

        '''
        Define the LSTM layer
        
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        '''
        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            dropout=0.5,
                            bidirectional=False,
                            batch_first=False)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # in_ = input.view(len(input), -1)
        embeds = self.embeddings(input)

        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        lstm_out, self.hidden = self.lstm(embeds)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1, :, :])
        return y_pred.view(-1)

    def fit(self, data_loader, val_data_loader, num_epochs, loss_fn, optimiser):
        for epoch in range(num_epochs):
            # ??
            self.train()

            hist = []
            pbar = tqdm(enumerate(data_loader))
            for i, batch in pbar:

                # Clear stored gradient
                self.zero_grad()
                text, target = batch.review, batch.label

                # Initialise hidden state
                # Don't do this if you want your LSTM to be stateful
                self.hidden = self.init_hidden()

                # Forward pass
                y_pred = self(text)

                target = target.float().view(-1)
                loss = loss_fn(y_pred, target)

                hist.append(loss.item())

                # Zero out gradient, else they will accumulate between epochs
                optimiser.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimiser.step()

                pbar.set_description('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, np.average(hist)))

            test_preds = []
            golden_preds = []
            for batch in tqdm(val_data_loader):
                text, target = batch.review, batch.label

                preds = self(text)
                preds = preds.cpu().data.numpy()
                # the actual outputs of the model are logits, so we need to pass these values to the sigmoid function
                # preds = 1 / (1 + np.exp(-preds))
                test_preds = np.append(test_preds, preds)
                golden_preds = np.append(golden_preds, target.cpu())
                test_preds = np.hstack(test_preds)
                golden_preds = np.hstack(golden_preds)

            equals = np.sum(np.equal(test_preds.round(), golden_preds.round()))
            pprint('Acc: {:.4f}'.format(equals / test_preds.size))
