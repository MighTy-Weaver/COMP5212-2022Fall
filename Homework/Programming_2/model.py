from torch import nn


class MLP_Classifier(nn.Module):
    def __init__(self, input_dim=3072, out_dim=10, activation='relu'):
        super(MLP_Classifier, self).__init__()

        self.in_dim = input_dim
        self.out_dim = out_dim
        self.activation = activation
        if self.activation == 'relu':
            self.activation_function = nn.ReLU()
        elif self.activation == 'softmax':
            self.activation_function = nn.Softmax()
        elif self.activation == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        else:
            raise NotImplementedError("Activation function {} has not been implemented.".format(activation))

        self.MLP1 = nn.Sequential(nn.Linear(in_features=input_dim, out_features=1024), self.activation_function)
        self.MLP2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512), self.activation_function)
        self.MLP3 = nn.Sequential(nn.Linear(in_features=512, out_features=256), self.activation_function)
        self.MLP4 = nn.Sequential(nn.Linear(in_features=256, out_features=128), self.activation_function)
        self.MLP5 = nn.Sequential(nn.Linear(in_features=128, out_features=64), self.activation_function)
        self.MLP6 = nn.Sequential(nn.Linear(in_features=64, out_features=32), self.activation_function)
        self.MLP7 = nn.Sequential(nn.Linear(in_features=32, out_features=out_dim), self.activation_function)

    def forward(self, x):
        x1 = self.MLP1(x)
        x2 = self.MLP2(x1)
        x3 = self.MLP3(x2)
        x4 = self.MLP4(x3)
        x5 = self.MLP5(x4)
        x6 = self.MLP6(x5)
        x7 = self.MLP7(x6)
        return x7


class CNN_Classifier(nn.Module):
    def __init__(self, input_channel=3, output_channel=256, output_dim=10, activation='relu'):
        super(CNN_Classifier, self).__init__()

        self.in_channel = input_channel
        self.out_channel = output_channel
        self.out_dim = output_dim
        self.activation = activation
        if self.activation == 'relu':
            self.activation_function = nn.ReLU()
        elif self.activation == 'softmax':
            self.activation_function = nn.Softmax()
        elif self.activation == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        else:
            raise NotImplementedError("Activation function {} has not been implemented.".format(activation))

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=64, kernel_size=(3, 3), stride=1, padding=1))
        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1))
        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=2, padding=1))
        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=self.out_channel, kernel_size=(3, 3), stride=2, padding=1))
        self.MLP1 = nn.Sequential(nn.Linear(in_features=4096, out_features=1024), self.activation_function)
        self.MLP2 = nn.Sequential(nn.Linear(in_features=1024, out_features=1024), self.activation_function)
        self.MLP3 = nn.Sequential(nn.Linear(in_features=1024, out_features=10), self.activation_function)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(x1)
        x3 = self.Conv3(x2)
        x4 = self.Conv4(x3)
        x5 = self.MLP1(x4)
        x6 = self.MLP2(x5)
        x7 = self.MLP3(x6)
        return x7
