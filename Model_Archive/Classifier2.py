# class Classifier3(nn.Module):  # 98 percent error on all types of error
#     def __init__(self, in_features=1, h1_size=16, h2_size=32, h3_size=32):
#         super(Classifier3, self).__init__()
#         self.lstm = T.nn.LSTM(in_features, h1_size, 2, batch_first=True)
#
#         self.lin_1 = T.nn.Linear(h1_size, h2_size)
#         self.lin_2 = T.nn.Linear(h2_size, h3_size)
#         self.lin_out = T.nn.Linear(h3_size, 1)
#
#         T.nn.init.kaiming_normal_(self.lin_1.weight, mode='fan_in', nonlinearity='relu')
#         T.nn.init.kaiming_normal_(self.lin_2.weight, mode='fan_in', nonlinearity='relu')
#         T.nn.init.kaiming_normal_(self.lin_out.weight, mode='fan_in', nonlinearity='sigmoid')
#
#     def forward(self, X):
#         # encoder part
#         hns, _ = self.lstm(X)[1]
#
#         h1 = T.relu(self.lin_1(hns[-1]))
#         h2 = T.relu(self.lin_2(h1))
#         Y = T.sigmoid(self.lin_out(h2))
#         return Y