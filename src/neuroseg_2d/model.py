"""Define 2D U-Net for binary tumor segmentation from FLAIR slices.

Usage:
    python -m neuroseg_2d.model
"""

__author__ = "Vu Hoi HUYNH"

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from neuroseg_2d.constants import FEATURES


class DoubleConv(nn.Module):
    """Apply two convolution, normalization, and activation stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build one two-stage convolution block."""

        super().__init__()

        # Padding preserves spatial size; BatchNorm2d supplies the learned offset.
        self.conv = nn.Sequential(
            # First convolution stage.
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second convolution stage.
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform one feature map without changing its height or width."""
        return self.conv(inputs)


class UNet(nn.Module):
    """Segment one-channel FLAIR slices with a binary 2D U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = FEATURES,
    ) -> None:
        """Build encoder, bottleneck, decoder, and output layers."""

        super().__init__()

        if not features or any(feature <= 0 for feature in features):
            raise ValueError("Features must be a non-empty tuple of positive integers.")

        # Encoder: increase channels for more deep contexts, but also reduce spatial size.
        self.in_channels = in_channels
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck: process deepest features between encoder and decoder.
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: restore spatial size and merge matching encoder features.
        self.ups = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # Output: map decoded features to raw segmentation logits.
        self.final = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Return raw segmentation logits with the same spatial size as inputs."""

        if x.ndim != 4:
            raise ValueError(f"Expected input shaped (B, C, H, W), received {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, received {x.shape[1]}.")

        skip_connections: list[Tensor] = []

        # Encode and save feature maps before pooling for skip connections.
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Transform deepest feature map.
        x = self.bottleneck(x)

        # Decode from deepest to shallowest level.
        skip_connections.reverse()
        for idx in range(0, len(self.ups), 2):
            # Upsample decoder feature map.
            x = self.ups[idx](x)
            # Merge matching encoder features to restore spatial context.
            skip_connection = skip_connections[idx // 2]
            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            concatenated = torch.cat((x, skip_connection), dim=1)
            # Decode concatenated features.
            x = self.ups[idx + 1](concatenated)

        return self.final(x)


def main() -> None:
    """Print default U-Net architecture."""

    model = UNet()
    print(model)


if __name__ == "__main__":
    main()
