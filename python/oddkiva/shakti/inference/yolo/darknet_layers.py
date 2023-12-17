import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (F.tanh(F.softplus(x)))
        return x


class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # Let's use shortcut variables.
        s = self.stride
        # Get the height and width of the input signal.
        h, w = x.shape[2:]

        # The kernel radius is calculated as
        r = self.kernel_size // 2

        # We calculate the easy part of the padding.
        p_left = r - 1 if self.kernel_size % 2 == 0 else r
        p_top = r - 1 if self.kernel_size % 2 == 0 else r

        # Now moving on the trickiest part of the padding.
        #
        # The max pool layers collects (w // s) x (h // s) samples from the
        # input signal.
        #
        # If we reason in 1D, the samples are located at:
        #   0, s, 2s, 3s, ... , (w // s - 1) * s
        #
        # The input signal is extended spatially so that it contains the
        # following sample points.
        x_last = (w // s - 1) * s + r
        y_last = (h // s - 1) * s + r

        # Therefore the last two padding are
        p_right = 0 if x_last == w - 1 else x_last - w + 1
        p_bottom = 0 if y_last == h - 1 else y_last - h + 1

        # Apply the padding.
        x_zero_padded = F.pad(x, (p_left, p_right, p_top, p_bottom),
                              mode='constant', value=0)

        # Apply the spatial max-pool function.
        y = F.max_pool2d(x_zero_padded, self.size, self.stride)

        return y
