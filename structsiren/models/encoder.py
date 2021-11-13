import typing

from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


class EfficientNetEncoder(torch.nn.Module):
    def __init__(
            self,
            num_codes: int,
            size_per_code: int,
            *,
            hidden_dims: typing.Union[int, typing.List] = [],
            name: str = 'efficientnet-b0',
            pretrained: bool = True
    ):
        super().__init__()
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained(name)
        else:
            self.efficientnet = EfficientNet.from_name(name)

        self.num_codes = num_codes
        self.size_per_code = size_per_code
        embed_dim = num_codes * size_per_code
        d_0 = self.efficientnet._conv_head.out_channels  # EfficientNet output

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        linear = []
        d_in = d_0

        for hidden_dim in hidden_dims:
            linear.append(
                nn.Linear(d_in, hidden_dim)
            )
            d_in = hidden_dim

        linear.append(nn.Linear(d_in, embed_dim))
        self.linear = nn.Sequential(*linear)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        r"""Extract features with EfficientNet.

        Shapes:
            x: (N, H, W, C)
            output: (N, D)

        """
        img = self.efficientnet.extract_features(img)
        img = self.efficientnet._avg_pooling(img).squeeze()
        return self.linear(img)
