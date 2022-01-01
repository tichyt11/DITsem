# class LSTMAE0(nn.Module):  # best model so far
#     def __init__(self, in_features=1, h_size=120, h2_size=90):
#         super(LSTMAE0, self).__init__()
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
#         self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h_size, h2_size)
#         self.lin_c = T.nn.Linear(h_size, h2_size)
#
#         self.lin = T.nn.Linear(h2_size, in_features)
#
#     def forward(self, X):
#         # encoder part
#         # all_h_en, _ = self.lstmEnFull(X)  # take all hidden states
#         (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin(h_de)
#         return Y
#
#
# class LSTMAE0n(nn.Module):  # LSTMAE0 with dropout and normalized weights
#     def __init__(self, in_features=1, h_size=90, h2_size=80):
#         super(LSTMAE0n, self).__init__()
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True, dropout=0.2)
#         self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
#         self.lin_h = T.nn.Linear(h_size, h2_size)
#         self.lin_c = T.nn.Linear(h_size, h2_size)
#         self.lin_out = T.nn.Linear(h2_size, in_features)
#
#         self.dropout_h = T.nn.Dropout2d(0)
#         self.dropout_c = T.nn.Dropout2d(0)
#         self.dropout_out = T.nn.Dropout2d(0)
#
#         T.nn.init.kaiming_normal_(self.lin_h.weight, mode='fan_in', nonlinearity='linear')
#         T.nn.init.kaiming_normal_(self.lin_c.weight, mode='fan_in', nonlinearity='linear')
#         T.nn.init.kaiming_normal_(self.lin_out.weight, mode='fan_in', nonlinearity='linear')
#
#     def forward(self, X):
#         # encoder part
#         (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(self.dropout_h(h_en[-1]))), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(self.dropout_c(c_en[-1]))), dim=0)
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(self.dropout_out(h_de))
#         return Y
#
#
# class LSTMAE_new(nn.Module):  # LSTMAE0, but output is only 1 sequence
#     def __init__(self, in_features=1, h_size=30, h2_size=40):
#         super(LSTMAE_new, self).__init__()
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
#         self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h_size, h2_size)
#         self.lin_c = T.nn.Linear(h_size, h2_size)
#
#         self.lin_out = T.nn.Linear(h2_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y
#
#
# class LSTMAE_new2(nn.Module):  # LSTMAE0, but output is only 1 sequence
#     def __init__(self, in_features=1, h_size=20, h2_size=30):
#         super(LSTMAE_new2, self).__init__()
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
#         self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h_size, h2_size)
#         self.lin_c = T.nn.Linear(h_size, h2_size)
#
#         self.lin_out = T.nn.Linear(h2_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y
#
#
# class LSTMAE_new3(nn.Module):  # LSTMAE0, but output is only 1 sequence
#     def __init__(self, in_features=1, h_size=10, h2_size=30):
#         super(LSTMAE_new3, self).__init__()
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
#         self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h_size, h2_size)
#         self.lin_c = T.nn.Linear(h_size, h2_size)
#
#         self.lin_out = T.nn.Linear(h2_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y
#
#
# class LSTMAE_new4(nn.Module):  # LSTMAE0, but output is only 1 sequence
#     def __init__(self, in_features=1, h_size=5, h2_size=20):
#         super(LSTMAE_new4, self).__init__()
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
#         self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h_size, h2_size)
#         self.lin_c = T.nn.Linear(h_size, h2_size)
#
#         self.lin_out = T.nn.Linear(h2_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y
#
#
# class LSTMAE_new5(nn.Module):  # LSTMAE0, but output is only 1 sequence
#     def __init__(self, in_features=1, h_size=3, h2_size=20):
#         super(LSTMAE_new5, self).__init__()
#         self.lstmEnFull = T.nn.LSTM(in_features, h_size, 2, batch_first=True)
#         self.lstmDeFull = T.nn.LSTM(h_size, h2_size, 1, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h_size, h2_size)
#         self.lin_c = T.nn.Linear(h_size, h2_size)
#
#         self.lin_out = T.nn.Linear(h2_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en, c_en) = self.lstmEnFull(X)[1]  # take last hidden s
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y
#
#
# class LSTMAE_dual(nn.Module):
#     def __init__(self, in_features=1, h_size=2, h2_size=20):
#         super(LSTMAE_dual, self).__init__()
#         self.lstmEn1 = T.nn.LSTM(1, h_size, 2, batch_first=True)
#         self.lstmEn2 = T.nn.LSTM(in_features - 1, 9*h_size, 2, batch_first=True)
#
#         self.lstmDeFull = T.nn.LSTM(10*h_size, h2_size, 1, batch_first=True)
#
#         self.lin_h = T.nn.Linear(10*h_size, h2_size)
#         self.lin_c = T.nn.Linear(10*h_size, h2_size)
#
#         self.lin_out = T.nn.Linear(h2_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en1, c_en1) = self.lstmEn1(T.unsqueeze(X[:, :, 0], 2))[1]  # this takes in the first sensor data
#         (h_en2, c_en2) = self.lstmEn2(X[:, :, 1:])[1]  # this takes in all the other sensors data
#
#         h_en = T.cat((h_en1, h_en2), 2)  # stack output features together
#         c_en = T.cat((c_en1, c_en2), 2)
#
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)  # take the last hidden layer and repeat
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y
#
#
# class LSTMAE_dual2(nn.Module):
#     def __init__(self, in_features=1, h1_size=3, h2_size=27, h3_size=50):
#         super(LSTMAE_dual2, self).__init__()
#         self.lstmEn1 = T.nn.LSTM(1, h1_size, 2, batch_first=True)
#         self.lstmEn2 = T.nn.LSTM(in_features - 1, h2_size, 2, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h1_size + h2_size, h3_size)
#         self.lin_c = T.nn.Linear(h1_size + h2_size, h3_size)
#
#         self.lstmDeFull = T.nn.LSTM(h1_size + h2_size, h3_size, 1, batch_first=True)
#
#         self.lin_out = T.nn.Linear(h3_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en1, c_en1) = self.lstmEn1(T.unsqueeze(X[:, :, 0], 2))[1]  # this takes in the first sensor data
#         (h_en2, c_en2) = self.lstmEn2(X[:, :, 1:])[1]  # this takes in all the other sensors data
#
#         h_en = T.cat((h_en1, h_en2), 2)  # stack output features together
#         c_en = T.cat((c_en1, c_en2), 2)
#
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)  # take the last hidden layer and repeat
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y
#
#
# class LSTMAE_dual3(nn.Module):
#     def __init__(self, in_features=1, h1_size=3, h2_size=60, h3_size=60):
#         super(LSTMAE_dual3, self).__init__()
#         self.lstmEn1 = T.nn.LSTM(1, h1_size, 2, batch_first=True)
#         self.lstmEn2 = T.nn.LSTM(in_features - 1, h2_size, 2, batch_first=True)
#
#         self.lin_h = T.nn.Linear(h1_size + h2_size, h3_size)
#         self.lin_c = T.nn.Linear(h1_size + h2_size, h3_size)
#
#         self.lstmDeFull = T.nn.LSTM(h1_size + h2_size, h3_size, 1, batch_first=True)
#
#         self.lin_out = T.nn.Linear(h3_size, 1)
#
#     def forward(self, X):
#         # encoder part
#         (h_en1, c_en1) = self.lstmEn1(T.unsqueeze(X[:, :, 0], 2))[1]  # this takes in the first sensor data
#         (h_en2, c_en2) = self.lstmEn2(X[:, :, 1:])[1]  # this takes in all the other sensors data
#
#         h_en = T.cat((h_en1, h_en2), 2)  # stack output features together
#         c_en = T.cat((c_en1, c_en2), 2)
#
#         repeated_h_en = T.transpose(h_en[-1].repeat(X.size(1), 1, 1), 0, 1)  # take the last hidden layer and repeat
#
#         h_0 = T.unsqueeze(T.tanh(self.lin_h(h_en[-1])), dim=0)
#         c_0 = T.unsqueeze(T.tanh(self.lin_c(c_en[-1])), dim=0)
#
#         # decoder part
#         h_de, _ = self.lstmDeFull(repeated_h_en, (h_0, c_0))  # pass all hidden states of decoder
#         Y = self.lin_out(h_de)
#         return Y