import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """

        # take the log loss of the softmax of our input
        return torch.nn.NLLLoss()(torch.nn.LogSoftmax(dim=-1)(input), target)

class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # input size is 3*64*64
        # output size is 6 (num classes)
        self.linear = torch.nn.Linear(3*64*64, 6)

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        # let x be of dimensions (B, 3*64*64)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = torch.nn.Linear(3*64*64, 64)
        torch.nn.init.normal_(self.linear1.weight, std=.01)
        torch.nn.init.normal_(self.linear1.bias, std=.01)
        self.linear2 = torch.nn.Linear(64, 6)


    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        x = x.view(x.size(0), -1)
        x = self.linear2(torch.relu(self.linear1(x)))
        return x


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
