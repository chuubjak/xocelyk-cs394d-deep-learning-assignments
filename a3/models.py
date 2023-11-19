import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128], n_input_channels=3, n_output_channels=6, kernel_size=3):
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
            L.append(torch.nn.Conv2d(c, l, conv_kernel_size, stride=1, padding=conv_kernel_size//2))
            L.append(torch.nn.MaxPool2d((2, 2)))
            L.append(torch.nn.BatchNorm2d(l))
            c = l
        L.append(torch.nn.Conv2d(c, n_output_channels, kernel_size=1))
        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """         
        return self.network(x).mean(dim=(2, 3))

class Block(torch.nn.Module):
    def __init__(self, n_input_channels, n_output_channels, stride=1, dropout=0.5):
            super().__init__()
            self.network = torch.nn.Sequential(torch.nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                                            # add dropout
                                            # torch.nn.Dropout(dropout),
                                            torch.nn.BatchNorm2d(n_output_channels),
                                            torch.nn.MaxPool2d((2, 2)),
                                            torch.nn.ConvTranspose2d(n_output_channels, n_output_channels, kernel_size=2, padding=0, stride=2, bias=False),
                                            torch.nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                                            # add dropout
                                            # torch.nn.Dropout(dropout),
                                            torch.nn.BatchNorm2d(n_output_channels),
                                            torch.nn.MaxPool2d((2, 2)),
                                            torch.nn.ConvTranspose2d(n_output_channels, n_output_channels, kernel_size=2, padding=0, stride=2, bias=False))
            self.downsample = None
            if n_input_channels != n_output_channels or stride != 1:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input_channels, n_output_channels, kernel_size=1, stride=stride, bias=False), torch.nn.BatchNorm2d(n_output_channels))

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.network(x) + identity
    

class FCN(torch.nn.Module):
    def __init__(self, layers=[32, 64, 128], n_input_channels=3, n_output_channels=5):
        super().__init__()
        L = [torch.nn.Conv2d(n_input_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU()]
        c = 16
        for l in layers:
            L.append(Block(c, l, stride=1))
            c = l
        L.append(torch.nn.Conv2d(c, n_output_channels, kernel_size=1))
        # increase width by factor of 2, height by factor of 2
        L.append(torch.nn.ConvTranspose2d(n_output_channels, n_output_channels, kernel_size=8, stride=2, padding=3))
        L.append(torch.nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, stride=1, padding=1))
        self.network = torch.nn.Sequential(*L)
    
    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,5,64,64))
        """         
        # pad to 64 x 64
        x_width = None
        if x.shape[3] < 64:
            x_width = x.shape[3]
            x = F.pad(x, (0, 64 - x.shape[3], 0, 0))
        x_height = None
        if x.shape[2] < 64:
            x_height = x.shape[2]
            x = F.pad(x, (0, 0, 0, 64 - x.shape[2]))

        out = self.network(x)

        if x_width is not None:
            out = out[:, :, :, :x_width]
        if x_height is not None:
            out = out[:, :, :x_height, :]

        return out

model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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

