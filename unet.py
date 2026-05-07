"""
Advanced U-Net Architecture for Chest X-Ray Segmentation
Supports multi-class segmentation: background, left lung, right lung, heart
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks — the core U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, dropout: float = 0.0):
        super().__init__()
        mid_ch = mid_ch or out_ch
        layers = [
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Encoder step: MaxPool → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Decoder step: Upsample → concat skip → DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if spatial dims differ
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Soft attention gate for focusing on relevant features."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Resize g1 to match x1 spatially
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    Advanced U-Net for Chest X-Ray segmentation.

    Args:
        in_channels:  1 for grayscale X-ray
        num_classes:  4 → [background, left_lung, right_lung, heart]
        features:     channel sizes at each encoder level
        bilinear:     use bilinear upsampling (True) or transposed conv (False)
        dropout:      dropout rate in conv blocks (helps regularisation)
        use_attention: add attention gates in decoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        features: list = [64, 128, 256, 512],
        bilinear: bool = True,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        # ── Encoder ──────────────────────────────────────────────────────────
        self.inc = DoubleConv(in_channels, features[0])
        self.downs = nn.ModuleList(
            [Down(features[i], features[i + 1], dropout) for i in range(len(features) - 1)]
        )

        # ── Bottleneck ────────────────────────────────────────────────────────
        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2 // factor, dropout=dropout)

        # ── Decoder ───────────────────────────────────────────────────────────
        rev = list(reversed(features))
        self.ups = nn.ModuleList()
        self.attns = nn.ModuleList()
        ch = features[-1] * 2 // factor
        for feat in rev:
            self.ups.append(Up(ch + feat, feat if bilinear else feat, bilinear, dropout))
            if use_attention:
                self.attns.append(AttentionGate(ch, feat, feat // 2))
            ch = feat

        # ── Output heads ──────────────────────────────────────────────────────
        self.seg_head = nn.Conv2d(features[0], num_classes, 1)   # segmentation
        self.depth_head = nn.Sequential(                          # depth map
            nn.Conv2d(features[0], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse

        for i, up in enumerate(self.ups):
            skip = skip_connections[i]
            if self.use_attention:
                skip = self.attns[i](x, skip)
            x = up(x, skip)

        seg_out = self.seg_head(x)
        depth_out = self.depth_head(x)
        return seg_out, depth_out


# ── Depth-only head (used in inference when only depth is needed) ──────────────
class DepthEstimator(nn.Module):
    """Lightweight decoder on top of a frozen U-Net encoder for fast depth prediction."""

    def __init__(self, encoder_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(encoder_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.net(features)


def get_model(device: str = "cpu") -> UNet:
    model = UNet(
        in_channels=1,
        num_classes=4,
        features=[64, 128, 256, 512],
        bilinear=True,
        dropout=0.15,
        use_attention=True,
    )
    return model.to(device).eval()
