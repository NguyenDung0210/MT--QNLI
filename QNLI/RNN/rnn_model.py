import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout):
        super(RNNModel, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input_ids, hidden):
        batch_size = input_ids.size(0)
        input_ids = input_ids.long()
        embeds = self.embedding(input_ids)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out.view(batch_size, -1, self.output_size)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        num_directions = 2 if self.lstm.bidirectional else 1
        hidden = (weight.new(self.n_layers*num_directions, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers*num_directions, batch_size, self.hidden_dim).zero_().to(device))
        return hidden