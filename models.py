from torch import nn
import torch as T
from torch.nn.functional import relu

T.manual_seed(1)

#TODO: add parameter normalization as in VIR sem

#TODO: create suitable architecture
# ideas: 1) sequence predictor and then evaluate error
#        2) auto-encoder and evaluate error on whole sequence
#        3) train an end-to-end classifier outputing if a fault has occured - harder dataset generation
#        4) encode using lstm and then use e.g. classifier, or clustering for outlier detection - train for minimal distance
#           in fault-free data and max distance in faulty data
#        5) is it possible to use convolution?
#           e.g. on all encoder hidden states into all decoder hidden states


class LSTMAE2(nn.Module):
    def __init__(self, in_features=1, h_size=10, h2_size=10):
        super(LSTMAE2, self).__init__()
        # self.lstmEn = T.nn.LSTMCell(in_features, h_size)
        self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
        self.lstmDeFull = T.nn.LSTM(h2_size, in_features, 1, batch_first=True)

        # self.lstmDe = T.nn.LSTMCell(h_size, h2_size)
        # self.lin = T.nn.Linear(h2_size + in_features, in_features, bias=False)
        self.lin = T.nn.Linear(h2_size, h2_size)
        self.lin2 = T.nn.Linear(h2_size, in_features)

    def forward(self, X):
        # encoder part
        # h_en, c_en = self.lstmEn(X[:, 0, :])  # initialization of the hidden state and cell state
        # for i in range(1, X.size()[1]):  # for all next time steps compute the hidden state
        #     h_en, c_en = self.lstmEn(X[:, i, :], (h_en, c_en))

        h_en = self.lstmEnFull(X)[0][:, -1, :]  # should work the same as the cell lstm
        # h_en = self.lstmEnFull(X)[1][0][1]  # should work the same as the cell lstm

        # decoder part
        Y = T.empty(X.size(), device='cuda', dtype=X.dtype)
        h_de, c_de = self.lstmDe(h_en)  # pass the last hidden state of encoder to decoder
        inter = relu(self.lin(h_de))
        # Y[:, 0, :] = h_de
        Y[:, 0, :] = self.lin2(inter)
        print(Y.size())

        for i in range(1, X.size()[1]):  # for all next time steps compute the hidden state
            h_de, c_de = self.lstmDe(h_en, (h_de, c_de))
            # Y[:, i, :] = self.lin(T.cat((h_en, h_de), 1))  # add hidden of encoder and decoder and pass through linear
            inter = relu(self.lin(h_de))
            Y[:, i, :] = self.lin2(inter)  # add hidden of encoder and decoder and pass through linear
        return Y


class LSTMAE0(nn.Module):
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


class LSTMAE(nn.Module):
    def __init__(self, in_features=1, h_size=150, h2_size=110):
        super(LSTMAE, self).__init__()
        self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True, dropout=0.2)
        self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
        self.lin_h = T.nn.Linear(h_size, h2_size)
        self.lin_c = T.nn.Linear(h_size, h2_size)
        self.lin_out = T.nn.Linear(h2_size, in_features)

        self.dropout_h = T.nn.Dropout2d(0.1)
        self.dropout_c = T.nn.Dropout2d(0.1)
        self.dropout_out = T.nn.Dropout2d(0.1)

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


if __name__=='__main__':

    device = "cuda" if T.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = LSTMAE().to(device)
    print(model)