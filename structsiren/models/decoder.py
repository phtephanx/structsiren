import numpy as np
import torch
import torch.nn as nn

from .siren import SineLayer
from .utils import get_meshgrid


class SirenStrTfm(nn.Module):
    r"""Module to transform implicit representations with latent codes.

    The latent code is mapped to bias and multiplicand.

    Args:
        size_per_code: number of dimensions for each code
        hidden_dim: hidden dimension of output of SIREN layer

    """
    def __init__(
            self,
            size_per_code: int,
            hidden_dim: int
    ):
        super().__init__()
        self.q_transform = nn.Linear(
            in_features=size_per_code,
            out_features=2 * hidden_dim,  # bias and multiplicand
            bias=True
        )
        self.dim_per_factor = size_per_code
        self.hidden_dim = hidden_dim

    def forward(
            self,
            x_i: torch.Tensor,
            q_i: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x_i: implicit representation
            q_i: latent code

        Returns:

        Shape:
            x_i: (n, h*w*c, coord-dim)
            q_i: (n, dim_per_factor)

        """
        n, h = x_i.shape[0], self.hidden_dim
        q_i = self.q_transform(q_i)
        amp = q_i[:, :h].reshape(n, 1, h)
        bias = q_i[:, h:].reshape(n, 1, h)
        return amp * x_i + bias


class StructSirenDecoder(nn.Module):
    r"""Structural Siren Decoder.

    Args:
        num_codes: number of codes with one code corresponding to one factor
        size_per_code: number of dimensions assigned to each code
        h: height of image to reconstruct
        w: width of image to reconstruct
        num_channels: number of channels to reconstruct
        siren_hidden_dim: hidden dimensions of siren layers
        sigmoid: apply sigmoid to output
        outermost_linear: option to apply linear layer instead of siren layer
            as last layer
        first_omega_0: frequency of first siren layer
        hidden_omega_0: frequencies of hidden siren layers

    Shape:
        input: (N, D)
        output: (N, 1, F, S)

    """
    def __init__(
        self,
        num_codes: int,
        size_per_code: int,
        h: int,
        w: int,
        num_channels: int,
        *,
        siren_hidden_dim: int = 128,
        sigmoid: bool = False,
        outermost_linear: bool = False,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
    ):
        super().__init__()
        self.size_per_code = size_per_code
        self.num_codes = num_codes
        self.sigmoid = sigmoid
        self.h = h
        self.w = w
        self.num_channels = num_channels
        self.x = get_meshgrid((h, w))  # shape: x = (F*S, 2)
        grid_dim = self.x.shape[-1]

        self.scm = nn.ModuleList(
            [
                SineLayer(
                    in_features=grid_dim,
                    out_features=siren_hidden_dim,
                    is_first=True,
                    omega_0=first_omega_0
                )
            ]
        )

        for _ in range(num_codes):
            self.scm.append(
                 nn.ModuleList(
                     [
                         SirenStrTfm(
                             size_per_code=size_per_code,
                             hidden_dim=siren_hidden_dim,
                         ),
                         SineLayer(
                             siren_hidden_dim,
                             siren_hidden_dim,
                             is_first=True,
                             omega_0=first_omega_0
                         )
                     ]
                 )
            )

        if outermost_linear:
            final_linear = nn.Linear(siren_hidden_dim, num_channels)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / siren_hidden_dim) / hidden_omega_0,
                    np.sqrt(6 / siren_hidden_dim) / hidden_omega_0
                )

            self.scm.append(final_linear)
        else:
            self.scm.append(SineLayer(siren_hidden_dim, num_channels,
                                      is_first=False, omega_0=hidden_omega_0))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q: latent code vector

        Shapes:
            q: (N, num_factors * dim_per_factor)
            x: (N, seq_len*num_features, coordinate_dim)

        Returns:
            reconstructed input

        """
        device = q.device
        n = q.shape[0]

        # split code vector into multiple codes
        q = q.chunk(chunks=self.num_codes, dim=-1)

        # allows to take derivative w.r.t. input
        x = self.x.to(device).clone().detach().requires_grad_(True)
        x = x.repeat(n, 1, 1)

        siren = self.scm[0]
        x = siren(x)

        for i in range(1, self.num_codes + 1):
            strtfm, siren = self.scm[i]

            x = strtfm(x, q[i-1])
            x = siren(x)

        x = self.scm[-1](x)

        if self.sigmoid:
            x = torch.sigmoid(x)

        return x.reshape(n, self.num_channels, self.h, self.w)


if __name__ == '__main__':
    ss = StructSirenDecoder(6, 2, 64, 64, 3)
    q = torch.randn(1, 12)
    print(ss(q).shape)
