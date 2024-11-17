import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, num_frequencies):
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, x):
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)
