import torch as T
from torch import nn


# did not work very well
class LSTMAE_01(nn.Module):  # LSTMAE with added linear into first LSTM h and c and a bigger encoder
    def __init__(self, in_features=1, h_size=200, h2_size=80, lstm_l1=2, lstm_l2=1):
        super(LSTMAE_01, self).__init__()

        self.lstm_l1 = lstm_l1
        self.lstm_l2 = lstm_l2
        self.h_size = h_size
        self.h2_size = h2_size

        self.lin_h1 = T.nn.Linear(in_features, lstm_l1*h_size)
        self.lin_c1 = T.nn.Linear(in_features, lstm_l1*h_size)
        self.lstmEnFull = T.nn.LSTM(in_features, h_size, lstm_l1, batch_first=True, dropout=0.2)
        self.lin_h2 = T.nn.Linear(h_size, h2_size)
        self.lin_c2 = T.nn.Linear(h_size, h2_size)
        self.lstmDeFull = T.nn.LSTM(h_size, h2_size, lstm_l2, batch_first=True)
        self.lin_out = T.nn.Linear(h2_size, in_features)

        self.dropout_h = T.nn.Dropout2d(0.2)
        self.dropout_c = T.nn.Dropout2d(0.2)
        self.dropout_out = T.nn.Dropout2d(0.2)

        T.nn.init.kaiming_normal_(self.lin_h1.weight, mode='fan_in', nonlinearity='tanh')
        T.nn.init.kaiming_normal_(self.lin_c1.weight, mode='fan_in', nonlinearity='tanh')
        T.nn.init.kaiming_normal_(self.lin_h2.weight, mode='fan_in', nonlinearity='tanh')
        T.nn.init.kaiming_normal_(self.lin_c2.weight, mode='fan_in', nonlinearity='tanh')
        T.nn.init.kaiming_normal_(self.lin_out.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, X):
        # encoder part
        h_10 = T.tanh(self.lin_h1(X[:, 0, :])).view(-1, self.lstm_l1, self.h_size).transpose(0, 1).contiguous()
        c_10 = T.tanh(self.lin_c1(X[:, 0, :])).view(-1, self.lstm_l1, self.h_size).transpose(0, 1).contiguous()
        (h_en, c_en) = self.lstmEnFull(X, (h_10, c_10))[1]  # take last hidden s
        repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
        # decoder part
        h_20 = T.unsqueeze(T.tanh(self.lin_h2(self.dropout_h(h_en[-1]))), dim=0)
        c_20 = T.unsqueeze(T.tanh(self.lin_c2(self.dropout_c(c_en[-1]))), dim=0)
        h_de, _ = self.lstmDeFull(repeated_h_en, (h_20, c_20))  # pass all hidden states of decoder
        Y = self.lin_out(self.dropout_out(h_de))
        return Y