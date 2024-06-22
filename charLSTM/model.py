import torch
import torch.nn as nn
import torch.nn.functional as F


class ArabicPoetryLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, dropout=0.5):
        super(ArabicPoetryLSTM, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        # Convert input to one-hot encoding
        x_one_hot = F.one_hot(x, num_classes=self.lstm.input_size).float()
        # Pass through LSTM
        output, hidden = self.lstm(x_one_hot, hidden)
        # Apply dropout to the LSTM output
        output = self.dropout(output)
        # Pass through fully connected layer
        prediction = self.fc(output)
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device))