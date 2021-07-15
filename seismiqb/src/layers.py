""" Special layers to incapsulate attribute computation into neural network. """
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from ..batchflow.models.torch import ResBlock



class InstantaneousPhaseLayer(nn.Module):
    """ Instantaneous phase computation along depth axis.

    Parameters
    ----------
    inputs : torch.Tensor, optional.

    continuous : bool, optional
        Transform phase from (-pi, pi) to (-pi / 2, pi / 2) to make it continuous or not, by default False.
        Transformation: f(phi) = abs(phi) - pi / 2.
    """
    def __init__(self, inputs=None, continuous=False, **kwargs):
        super().__init__()
        self.continuous = continuous

    def _hilbert(self, x):
        """ Hilbert transformation. """
        N = x.shape[-1]
        fft = torch.fft.fft(x)

        h = torch.zeros(N, device=x.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2
        if x.ndim > 1:
            shape = [1] * x.ndim
            shape[-1] = N
            h = h.view(*shape)

        result = torch.fft.ifft(fft * h)
        return result

    def _angle(self, x):
        """ Compute angle of complex number. """
        res = torch.atan(x.imag / x.real)
        res[x.real == 0] = np.pi
        res = res % (2 * np.pi) - np.pi
        if self.continuous:
            res =  torch.abs(res) - np.pi / 2
        return res

    def forward(self, x):
        """ Forward pass. """
        x = self._hilbert(x)
        x = self._angle(x)
        return x

class MovingNormalizationLayer(nn.Module):
    """ Normalize tensor by mean/std in moving window.

    Parameters
    ----------
    inputs : torch.Tensor, optional

    window : tuple, optional
        window shape to compute statistics, by default (1, 1, 100).
    padding : str, optional
        'valid' or 'same, by default 'same'.
    fill_value : int, optional
        Value to fill constant regions with std=0, by default 0.
    """
    def __init__(self, inputs, window=(1, 1, 100), padding='same', fill_value=0):
        super().__init__()
        self.window = window
        self.padding = padding
        self.fill_value = fill_value
        self.ndim = inputs.ndim
        self.kernel = torch.ones((1, 1, *window), dtype=inputs.dtype, requires_grad=False).to(inputs.device)

    @autocast(enabled=False)
    def forward(self, x):
        """ Forward pass. """
        x = expand_dims(x)
        pad = [(w // 2, w - w // 2 - 1) for w in self.window]
        if self.padding == 'same':
            num = F.pad(x, (*pad[2], *pad[1], *pad[0], 0, 0, 0, 0))
        else:
            num = x
        n = np.prod(self.window)
        mean = F.conv3d(num, self.kernel / n)
        mean_2 = F.conv3d(num ** 2, self.kernel / n)
        std = torch.clip(mean_2 - mean ** 2, min=0) ** 0.5
        if self.padding == 'valid':
            x = x[:, :, pad[0][0]:x.shape[2]-pad[0][1], pad[1][0]:x.shape[3]-pad[1][1], pad[2][0]:x.shape[4]-pad[2][1]]
        result = torch.nan_to_num((x - mean) / std, nan=self.fill_value)
        return squueze(result, self.ndim)

class SemblanceLayer(nn.Module):
    """ Semblance attribute.

    Parameters
    ----------
    inputs : torch.Tensor, optional.

    window : tuple, optional
        Window shape to compute attribute, by default (1, 5, 20).
    fill_value : int, optional
        Value to fill constant regions, by default 1.
    """
    def __init__(self, inputs, window=(1, 5, 20), fill_value=1):
        super().__init__()
        self.ndim = inputs.ndim
        self.window = window
        self.fill_value = fill_value
        self.device = inputs.device

        self.kernels = [
            torch.ones((1, 1, window[0], window[1], 1), dtype=inputs.dtype, requires_grad=False).to(self.device),
            torch.ones((1, 1, 1, 1, window[2]), dtype=inputs.dtype, requires_grad=False).to(self.device),
            torch.ones((1, 1, *window), dtype=inputs.dtype, requires_grad=False).to(self.device),
            torch.ones((1, 1, window[0], window[1]), dtype=inputs.dtype, requires_grad=False).to(self.device)
        ]

    def forward(self, x):
        """ Forward pass. """
        window = self.window
        x = expand_dims(x)

        padding = [(w // 2, w - w // 2 - 1) for w in window]
        num = F.pad(x, (0, 0, *padding[1], *padding[0], 0, 0, 0, 0))
        num = F.conv3d(num, self.kernels[0]) ** 2

        num = F.pad(num, (*padding[2], 0, 0, 0, 0, 0, 0, 0, 0))
        num = F.conv3d(num, self.kernels[1])

        denum = F.pad(x, (*padding[2], *padding[1], *padding[0], 0, 0, 0, 0))
        denum = F.conv3d(denum ** 2, self.kernels[2])

        normilizing = torch.ones(x.shape[:-1], dtype=x.dtype, requires_grad=False).to(x.device)
        normilizing = F.pad(normilizing, (*padding[1], *padding[0], 0, 0, 0, 0))
        normilizing = F.conv2d(normilizing, self.kernels[3])

        denum *= normilizing.view(*normilizing.shape, 1)
        result = torch.nan_to_num(num / denum, nan=self.fill_value)

        return squueze(result, self.ndim)

class FrequenciesFilterLayer(nn.Module):
    """ Frequencies filter.

    Parameters
    ----------
    inputs : torch.Tensor, optional

    q : float, optional
         Left quantile, by default 0.1. The right quantile will be `1 - q`.
    window : int, optional
        Window width (corresponds to depth axis) to compute phases, by default 200.
    """
    def __init__(self, inputs=None, q=0.1, window=200):
        super().__init__()
        self.q = q
        self.window = window

    def forward(self, inputs):
        """ Forward pass. """
        inputs = inputs.view(-1, inputs.shape[-1])
        # TODO: remove disable after torch update
        sfft = torch.stft(inputs, self.window, return_complex=True) #pylint: disable=unexpected-keyword-arg
        q_ = int(sfft.shape[-2] * self.q)
        sfft[:, :q_] = 0
        sfft[:, -q_:] = 0
        return torch.istft(sfft, self.window).view(*inputs.shape)

class InputLayer(nn.Module):
    """ Input layer with possibility of instantaneous phase concatenation.

    Parameters
    ----------
    inputs : torch.Tensor

    normalization : bool, optional
        Normalize input or nor, by default False.
    phases : bool, optional
        Concat instantaneous phases to input or not, by default False.
    continuous : bool, optional
        Make phases continuous or not, by default False.
    window : int, optional
        Normalization window, by default 100
    base_block : torch.nn.Module, optional
        Inputs transformations block, by default ResBlock.
    """
    def __init__(self, inputs, normalization=False, phases=False, continuous=False,
                 window=100, base_block=ResBlock, **kwargs):
        super().__init__()
        self.normalization = normalization
        self.phases = phases
        if self.normalization:
            self.normalization_layer = MovingNormalizationLayer(inputs, window)
        if self.phases:
            self.phase_layer = InstantaneousPhaseLayer(continuous)
            phases_ = self.phase_layer(inputs)
            inputs = self._concat(inputs, phases_)
        self.base_block = base_block(inputs, **kwargs)

    def _concat(self, x, phases):
        x = torch.cat([x, phases], dim=1)
        return x

    def forward(self, x):
        """ Forward pass. """
        if self.normalization:
            x = self.normalization_layer(x)
            x = torch.clip(x,  -10, 10) # TODO: remove clipping
        if self.phases:
            phases = self.phase_layer(x)
            x = self._concat(x, phases)
        x = self.base_block(x)
        return x

def expand_dims(x):
    """ Make tensor 5D. """
    if x.ndim == 4:
        x = x.view(x.shape[0], 1, *x.shape[-3:])
    elif x.ndim == 3:
        x = x.view(1, 1, *x.shape)
    elif x.ndim == 2:
        x = x.view(1, 1, 1, *x.shape)
    return x

def squueze(x, ndim):
    """ Squeeze axes after :func:`~expand_dims`. """
    if ndim == 4:
        return x[:, 0]
    if ndim == 3:
        return x[0, 0]
    if ndim == 2:
        return x[0, 0, 0]
    return x
