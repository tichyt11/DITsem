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