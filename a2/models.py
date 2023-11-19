import torch

class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64], n_input_channels=3, kernel_size=3):
        super().__init__()
        L = []
        c = n_input_channels
        first = True
        for l in layers:
            if first:
                conv_kernel_size = 7
                first = False
            else:
                conv_kernel_size = kernel_size
            L.append(torch.nn.Conv2d(c, l, conv_kernel_size, stride=2, padding=1))
            L.append(torch.nn.MaxPool2d((2, 2)))
            c = l
        L.append(torch.nn.Conv2d(c, 6, kernel_size=1))
        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """         
        return self.network(x).mean(dim=(2, 3))

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
