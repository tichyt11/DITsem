# class Classifier(nn.Module):  # really good at linear error vs healthy data
#     def __init__(self, in_features=1, h1_size=12, h2_size=12, h3_size=12):
#         super(Classifier, self).__init__()
#         self.lstmdim = 2
#         self.lstm1 = T.nn.LSTM(1, h1_size, self.lstmdim, batch_first=True)
#         self.lstm2 = T.nn.LSTM(in_features - 1, 2*h1_size, self.lstmdim, batch_first=True)
#
#         self.lin_11 = T.nn.Linear(h1_size, h2_size)
#         self.lin_12 = T.nn.Linear(2*h1_size, h2_size)
#
#         self.lin_2 = T.nn.Linear(2*h2_size, h3_size)
#         self.lin_out = T.nn.Linear(h3_size, 1)
#
#         T.nn.init.kaiming_normal_(self.lin_11.weight, mode='fan_in', nonlinearity='relu')
#         T.nn.init.kaiming_normal_(self.lin_12.weight, mode='fan_in', nonlinearity='relu')
#         T.nn.init.kaiming_normal_(self.lin_2.weight, mode='fan_in', nonlinearity='relu')
#         T.nn.init.kaiming_normal_(self.lin_out.weight, mode='fan_in', nonlinearity='sigmoid')
#
#     def forward(self, X):
#         # encoder part
#         out1, _ = self.lstm1(T.unsqueeze(X[:, :, 0], 2))[1]
#         out2, _ = self.lstm2(X[:, :, 1:])[1]
#
#         h11 = T.relu(self.lin_11(out1[-1]))
#         h12 = T.relu(self.lin_12(out2[-1]))
#         h1 = T.cat((h11, h12), 1)
#
#         h2 = T.relu(self.lin_2(h1))
#
#         Y = T.sigmoid(self.lin_out(h2))
#         return Y