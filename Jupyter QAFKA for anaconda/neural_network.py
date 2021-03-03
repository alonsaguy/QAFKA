import torch
import torch.nn as nn
import torch.optim

class ONE_LAYER_NET(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ONE_LAYER_NET, self).__init__()
        self.input_size = input_size
        self.hidden_size = [self.input_size] + hidden_size

        modules = []
        modules.append(torch.nn.Linear(self.hidden_size[0], 1, bias=True))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        output = self.net(x)
        return output

class SIMPLE_FC_NET(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SIMPLE_FC_NET, self).__init__()
        self.input_size = input_size
        self.hidden_size = [self.input_size] + hidden_size

        modules = []

        for i in range(len(self.hidden_size) - 1):
            modules.append(torch.nn.Linear(self.hidden_size[i], self.hidden_size[i+1], bias=True))
            modules.append(torch.nn.PReLU())

        modules.append(torch.nn.Linear(self.hidden_size[-1], 1))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        output = self.net(x)
        return output

class CustomNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = [self.input_size] + hidden_size

        modules = []

        for i in range(len(self.hidden_size) - 1):
            modules.append(torch.nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))
            modules.append(torch.nn.PReLU())

        modules.append(torch.nn.Linear(self.hidden_size[-1], 1))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        output = self.net(x)
        return output

def BiasTrick(X_train, X_val, X_test):
    coef = 0.05
    ones = torch.ones((X_train.shape[0], 1)).type(torch.FloatTensor) * coef
    X_train = torch.cat((ones, X_train.type(torch.FloatTensor)), dim=1)
    ones = torch.ones((X_val.shape[0], 1)).type(torch.FloatTensor) * coef
    X_val = torch.cat((ones, X_val.type(torch.FloatTensor)), dim=1)
    ones = torch.ones((X_test.shape[0], 1)).type(torch.FloatTensor) * coef
    X_test = torch.cat((ones, X_test.type(torch.FloatTensor)), dim=1)
    return X_train, X_val, X_test