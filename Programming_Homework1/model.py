from torch import nn
from torch import sigmoid
from torch.nn import Linear


class LogisticRegression_Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression_Classifier, self).__init__()
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class SVM_Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SVM_Classifier, self).__init__()
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
