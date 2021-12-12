from torch import nn
import torch as T
from torch.nn.functional import relu

T.manual_seed(1)

#TODO: create suitable architecture
# ideas: 1) sequence predictor and then evaluate error
#        2) LSTM auto-encoder and evaluate error on whole sequence
#        3) Transformer auto-encoder -||-
#        4) train an end-to-end classifier outputing if a fault has occured - harder dataset generation
#        5) encode using lstm and then use e.g. classifier, or clustering for outlier detection - train for minimal distance
#           in fault-free data and max distance in faulty data
#        6) is it possible to use convolution?
#           e.g. on all encoder hidden states into all decoder hidden states


# class LSTMAE2(nn.Module):  # original garbage model
#     def __init__(self, in_features=1, h_size=10, h2_size=10):
#         super(LSTMAE2, self).__init__()
#         # self.lstmEn = T.nn.LSTMCell(in_features, h_size)
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
#         self.lstmDeFull = T.nn.LSTM(h2_size, in_features, 1, batch_first=True)
#
#         # self.lstmDe = T.nn.LSTMCell(h_size, h2_size)
#         # self.lin = T.nn.Linear(h2_size + in_features, in_features, bias=False)
#         self.lin = T.nn.Linear(h2_size, h2_size)
#         self.lin2 = T.nn.Linear(h2_size, in_features)
#
#     def forward(self, X):
#         # encoder part
#         # h_en, c_en = self.lstmEn(X[:, 0, :])  # initialization of the hidden state and cell state
#         # for i in range(1, X.size()[1]):  # for all next time steps compute the hidden state
#         #     h_en, c_en = self.lstmEn(X[:, i, :], (h_en, c_en))
#
#         h_en = self.lstmEnFull(X)[0][:, -1, :]  # should work the same as the cell lstm
#         # h_en = self.lstmEnFull(X)[1][0][1]  # should work the same as the cell lstm
#
#         # decoder part
#         Y = T.empty(X.size(), device='cuda', dtype=X.dtype)
#         h_de, c_de = self.lstmDe(h_en)  # pass the last hidden state of encoder to decoder
#         inter = relu(self.lin(h_de))
#         # Y[:, 0, :] = h_de
#         Y[:, 0, :] = self.lin2(inter)
#         print(Y.size())
#
#         for i in range(1, X.size()[1]):  # for all next time steps compute the hidden state
#             h_de, c_de = self.lstmDe(h_en, (h_de, c_de))
#             # Y[:, i, :] = self.lin(T.cat((h_en, h_de), 1))  # add hidden of encoder and decoder and pass through linear
#             inter = relu(self.lin(h_de))
#             Y[:, i, :] = self.lin2(inter)  # add hidden of encoder and decoder and pass through linear
#         return Y


class TRAE(nn.Module):
    def __init__(self, in_features=1, h_size=30, h2_size=30):
        super(TRAE, self).__init__()
        self.En = nn.TransformerEncoderLayer(d_model=in_features, nhead=2, batch_first=True, dim_feedforward=h_size)
        self.hlin = nn.Linear(in_features, h_size)
        self.enlin = nn.Linear(in_features, h_size)
        self.De = nn.TransformerDecoderLayer(d_model=h_size, nhead=2, batch_first=True, dim_feedforward=h2_size)
        self.lin2 = T.nn.Linear(h2_size, 1)

    def forward(self, X):
        encoded = self.En(X)
        hidden = T.tanh(self.hlin(encoded))
        memory = T.tanh(self.enlin(encoded))
        decoded = self.De(hidden, memory)
        Y = self.lin2(decoded)
        return Y



class LSTMAE0(nn.Module):  # best model so far
    def __init__(self, in_features=1, h_size=120, h2_size=90):
        super(LSTMAE0, self).__init__()
        self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
        self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)

        self.lin_h = T.nn.Linear(h_size, h2_size)
        self.lin_c = T.nn.Linear(h_size, h2_size)

        self.lin = T.nn.Linear(h2_size, in_features)

    def forward(self, X):
        # encoder part
        # all_h_en, _ = self.lstmEnFull(X)  # take all hidden states
        (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
        repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)

        h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
        c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)

        # decoder part
        h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
        Y = self.lin(h_de)
        return Y


class LSTMAE_new(nn.Module):
    def __init__(self, in_features=1, h_size=30, h2_size=20):
        super(LSTMAE_new, self).__init__()
        self.lstmEnFull = T.nn.LSTM(in_features, h_size, 1, batch_first=True)
        self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)

        self.lin_h = T.nn.Linear(h_size, h2_size)
        self.lin_c = T.nn.Linear(h_size, h2_size)

        # self.lin = T.nn.Linear(h2_size, h2_size)
        self.lin2 = T.nn.Linear(h2_size, 1)

    def forward(self, X):
        # encoder part
        # all_h_en, _ = self.lstmEnFull(X)  # take all hidden states
        (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
        repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)

        h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
        c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)

        # decoder part
        h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
        # Y = T.tanh(self.lin(h_de))
        Y = self.lin2(h_de)
        return Y


class LSTMAE(nn.Module):
    def __init__(self, in_features=1, h_size=90, h2_size=80):
        super(LSTMAE, self).__init__()
        self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True, dropout=0.2)
        self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
        self.lin_h = T.nn.Linear(h_size, h2_size)
        self.lin_c = T.nn.Linear(h_size, h2_size)
        self.lin_out = T.nn.Linear(h2_size, in_features)

        self.dropout_h = T.nn.Dropout2d(0.2)
        self.dropout_c = T.nn.Dropout2d(0.2)
        self.dropout_out = T.nn.Dropout2d(0.2)

        T.nn.init.kaiming_normal_(self.lin_h.weight, mode='fan_in', nonlinearity='linear')
        T.nn.init.kaiming_normal_(self.lin_c.weight, mode='fan_in', nonlinearity='linear')
        T.nn.init.kaiming_normal_(self.lin_out.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, X):
        # encoder part
        (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
        repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
        h_0 = T.unsqueeze(T.tanh(self.lin_h(self.dropout_h(h_en[-1]))), dim=0)
        c_0 = T.unsqueeze(T.tanh(self.lin_c(self.dropout_c(c_en[-1]))), dim=0)
        # decoder part
        h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
        Y = self.lin_out(self.dropout_out(h_de))
        return Y


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


if __name__ == '__main__':

    device = "cuda" if T.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = LSTMAE().to(device)
    print(model)