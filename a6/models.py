import torch
import torch.nn.functional as F

try:
    from . import utils
except:
    import utils

def pad_and_mask(sequences, max_seq_len):
    padded_sequences = torch.zeros(len(sequences), max_seq_len, sequences[0].size(1))
    mask = torch.zeros(len(sequences), max_seq_len)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :seq.size(0), :] = seq
        mask[i, :seq.size(0)] = 1
    return padded_sequences, mask

class LanguageModel(object):
    def predict_all(self, some_text):
        """
        Given some_text, predict the likelihoods of the next character for each substring from 0..i
        The resulting tensor is one element longer than the input, as it contains probabilities for all sub-strings
        including the first empty string (probability of the first character)

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: torch.Tensor((len(utils.vocab), len(some_text)+1)) of log-probabilities
        """
        
        raise NotImplementedError('Abstract function LanguageModel.predict_all')

    def predict_next(self, some_text):
        """
        Given some_text, predict the likelihood of the next character

        :param some_text: A string containing characters in utils.vocab, may be an empty string!
        :return: a Tensor (len(utils.vocab)) of log-probabilities
        """
        res = self.predict_all(some_text)[:, -1]
        return res


class Bigram(LanguageModel):
    """
    Implements a simple Bigram model. You can use this to compare your TCN to.
    The bigram, simply counts the occurrence of consecutive characters in transition, and chooses more frequent
    transitions more often. See https://en.wikipedia.org/wiki/Bigram .
    Use this to debug your `language.py` functions.
    """

    def __init__(self):
        from os import path
        self.first, self.transition = torch.load(path.join(path.dirname(path.abspath(__file__)), 'bigram.th'))

    def predict_all(self, some_text):
        return torch.cat((self.first[:, None], self.transition.t().matmul(utils.one_hot(some_text))), dim=1)


class AdjacentLanguageModel(LanguageModel):
    """
    A simple language model that favours adjacent characters.
    The first character is chosen uniformly at random.
    Use this to debug your `language.py` functions.
    """

    def predict_all(self, some_text):
        prob = 1e-3*torch.ones(len(utils.vocab), len(some_text)+1)
        if len(some_text):
            one_hot = utils.one_hot(some_text)
            prob[-1, 1:] += 0.5*one_hot[0]
            prob[:-1, 1:] += 0.5*one_hot[1:]
            prob[0, 1:] += 0.5*one_hot[-1]
            prob[1:, 1:] += 0.5*one_hot[:-1]
        return (prob/prob.sum(dim=0, keepdim=True)).log()


class TCN(torch.nn.Module):
    class CausalConv1dBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
            super().__init__()
            self.padding = torch.nn.ConstantPad1d((dilation * (kernel_size - 1), 0), 0)
            self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
            # torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
            self.relu = torch.nn.ReLU()

            # Add a 1x1 convolution for the residual connection if the number of channels change
            if in_channels != out_channels:
                self.residual_conv = torch.nn.Conv1d(in_channels, out_channels, 1)
            else:
                self.residual_conv = None

        def forward(self, x):
            residual = x
            x = self.padding(x)
            x = self.conv1(x)
            x = self.relu(x)

            # Apply the residual connection
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            x = x + residual
            return x

    def __init__(self, layers=[8, 16, 32, 64], num_blocks=2, kernel_size=3):
        super().__init__()
        self.init_distribution = torch.nn.Parameter(torch.randn(len(utils.vocab)))

        c = len(utils.vocab)
        self.blocks = []
        # dilation > 1 is not working!
        total_dilation = 1
        for l in layers:
            for _ in range(num_blocks):
                # equivalent to one block with two convolutions, one from in_channels to out_channels and one from out_channels to out_channels
                block = self.CausalConv1dBlock(c, l, kernel_size=kernel_size, dilation=total_dilation)
                self.blocks.append(block)
                c = l
            # total_dilation *= 2
        self.blocks = torch.nn.Sequential(*self.blocks)
        self.classifier = torch.nn.Conv1d(c, len(utils.vocab), 1)

    def forward(self, x):
        input_length = x.shape[2]
        x = torch.nn.functional.pad(x, (0, 250 - x.shape[2]))
        x = self.classifier(self.blocks(x))
        x = x[:, :, :input_length]
        init_distr = self.init_distribution.view(1, -1, 1).repeat(x.shape[0], 1, 1)
        x = torch.cat([init_distr, x], dim=2)
        return x

    def predict_all(self, some_text):
        """
        Your code here

        @some_text: a string
        @return torch.Tensor((vocab_size, len(some_text)+1)) of log-likelihoods (not logits!)
        """
        some_text = utils.one_hot(some_text)[None, :, :]
        x = self.forward(some_text).squeeze(0)
        # normalize the log likelihoods over the second dimension
        x = x - x.logsumexp(dim=0, keepdim=True)
        return x
    
    def predict_next(self, some_text):
        return self.predict_all(some_text)[:, -1]

def save_model(model):
    from os import path
    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))


def load_model():
    from os import path
    r = TCN()
    r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'), map_location='cpu'))
    return r
