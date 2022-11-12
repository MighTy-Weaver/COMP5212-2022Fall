import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, num_layer=1, emb_dim=300, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.LSTM = nn.LSTM(input_size=emb_dim, hidden_size=512, num_layers=num_layer, bias=True, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.nn1 = nn.Sequential(nn.Linear(in_features=1024, out_features=256), nn.LeakyReLU())
        self.nn2 = nn.Sequential(nn.Linear(in_features=256, out_features=2))

    def forward(self, x):
        output, (h_n, c_n) = self.LSTM(x)
        output_mean = torch.mean(output, dim=1).squeeze()
        x2 = self.nn1(output_mean)
        return self.nn2(x2)
