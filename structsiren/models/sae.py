import typing

import torch

from .encoder import EfficientNetEncoder
from .decoder import StructSirenDecoder


class SirenSAE(torch.nn.Module):
    def __init__(
        self,
        encoder,
        decoder: StructSirenDecoder
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            img: torch.Tensor,
            *,
            return_codes: bool = False
    ) -> typing.Union[
        torch.Tensor,
        typing.Tuple[torch.Tensor, torch.Tensor]
    ]:
        """

        Args:
            img: image to decompose and reconstruct
            return_codes: option to return latent codes

        Shapes:
            * img: (N, H, W, C)

        """
        codes = self.encoder(img)
        reconstructed = self.decoder(codes)

        if return_codes:
            return reconstructed, codes

        return reconstructed


class EfficientNetSirenSAE(SirenSAE):
    @classmethod
    def build(
            cls,
            h_dim: int,
            w_dim: int,
            num_codes: int,
            size_per_code: int,
            *,
            num_channels: int = 3,
            encoder_hidden_dims: int = 1024,
            sigmoid: bool = True,
            name: str = 'efficientnet-b0',
            pretrained_encoder: bool = False,
            siren_hidden_dim: int = 128,
            first_omega_0: float = 30,
            hidden_omega_0: float = 30
    ):
        encoder = EfficientNetEncoder(
            num_codes=num_codes,
            size_per_code=size_per_code,
            hidden_dims=encoder_hidden_dims,
            name=name,
            pretrained=pretrained_encoder
        )
        decoder = StructSirenDecoder(
            num_codes=num_codes,
            size_per_code=size_per_code,
            h=h_dim,
            w=w_dim,
            num_channels=num_channels,
            siren_hidden_dim=siren_hidden_dim,
            outermost_linear=True,
            sigmoid=sigmoid,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0
        )
        return EfficientNetSirenSAE(
            encoder=encoder,
            decoder=decoder
        )
