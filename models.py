from torch import nn
import torch as T

T.manual_seed(1)


# this is the model used in the paper
class FullClassifier_best(nn.Module):  # 98 on training data
    def __init__(self, in_features=1, h1_size=20, h2_size=32, h3_size=32):
        super(FullClassifier_best, self).__init__()
        self.lin_h = T.nn.Linear(in_features, h1_size)
        self.lin_c = T.nn.Linear(in_features, h1_size)
        self.lstm = T.nn.LSTM(in_features, h1_size, 1, batch_first=True)

        self.lin_1 = T.nn.Linear(h1_size, h2_size)
        self.lin_2 = T.nn.Linear(h2_size, h3_size)
        self.lin_out = T.nn.Linear(h3_size, 3)

        T.nn.init.kaiming_normal_(self.lin_1.weight, mode='fan_in', nonlinearity='relu')
        T.nn.init.kaiming_normal_(self.lin_2.weight, mode='fan_in', nonlinearity='relu')
        T.nn.init.kaiming_normal_(self.lin_out.weight, mode='fan_in', nonlinearity='sigmoid')

    def forward(self, X):
        # encoder part
        h0 = T.tanh(self.lin_h(X[:, 0, :])).unsqueeze(0)
        c0 = T.tanh(self.lin_c(X[:, 0, :])).unsqueeze(0)
        hns, _ = self.lstm(X, (h0, c0))[1]

        h1 = T.relu(self.lin_1(hns[-1]))
        h2 = T.relu(self.lin_2(h1))
        Y = T.sigmoid(self.lin_out(h2))
        return Y


class Fclassifier_2(nn.Module):
    def __init__(self, in_features=1, h1_size=20, h2_size=32, h3_size=32):
        super(Fclassifier_2, self).__init__()
        self.lin_h = T.nn.Linear(in_features, h1_size)
        self.lin_c = T.nn.Linear(in_features, h1_size)
        self.lstm = T.nn.LSTM(in_features, h1_size, 1, batch_first=True)

        self.lin_1 = T.nn.Linear(h1_size, h2_size)
        self.lin_2 = T.nn.Linear(h2_size, h3_size)
        self.lin_out = T.nn.Linear(h3_size, 3)

        self.lrelu = T.nn.LeakyReLU()

        self.dropout_1 = T.nn.Dropout(0.2)
        self.dropout_2 = T.nn.Dropout(0.2)
        self.dropout_out = T.nn.Dropout(0.2)

        T.nn.init.kaiming_normal_(self.lin_1.weight, mode='fan_in', nonlinearity='relu')
        T.nn.init.kaiming_normal_(self.lin_2.weight, mode='fan_in', nonlinearity='relu')
        T.nn.init.kaiming_normal_(self.lin_out.weight, mode='fan_in', nonlinearity='sigmoid')

    def forward(self, X):
        h0 = T.tanh(self.lin_h(X[:, 0, :])).unsqueeze(0)
        c0 = T.tanh(self.lin_c(X[:, 0, :])).unsqueeze(0)
        hns, _ = self.lstm(X, (h0, c0))[1]

        h1 = self.lrelu(self.lin_1(self.dropout_1(hns[-1])))
        h2 = self.lrelu(self.lin_2(self.dropout_2(h1)))
        Y = T.sigmoid(self.lin_out(self.dropout_out (h2)))
        return Y


class Fclassifier_1(nn.Module):
    def __init__(self, in_features=1, h1_size=24, h2_size=32, h3_size=32):
        super(Fclassifier_1, self).__init__()
        self.lin_h = nn.Linear(in_features, h1_size)
        self.lin_c = nn.Linear(in_features, h1_size)
        self.lstm = nn.LSTM(in_features, h1_size, 1, batch_first=True)

        self.FCstage = nn.Sequential(
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
            nn.Linear(h2_size, h3_size),
            nn.ReLU(),
            nn.Linear(h3_size, 3),
            nn.Sigmoid()
        )

    def forward(self, X):
        h0 = T.tanh(self.lin_h(X[:, 0, :])).unsqueeze(0)
        c0 = T.tanh(self.lin_c(X[:, 0, :])).unsqueeze(0)
        hns, _ = self.lstm(X, (h0, c0))[1]
        Y = self.FCstage(hns[-1])
        return Y


if __name__ == '__main__':

    device = "cuda" if T.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = FullClassifier_best().to(device)
    print(model)