import math

from torch import nn

from ..utils.helpers import as_tuple, partialize


class DoubleConv(nn.Module):
    """(Conv -- Drop -- Norm -- Act) ** 2."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv = partialize(conv)
        drop = partialize(drop)
        norm = partialize(norm)
        act = partialize(act)

        self.block1 = nn.Sequential(
            conv(in_channels, mid_channels, stride=stride),
            drop(),
            norm(mid_channels),
            act(),
        )

        self.block2 = nn.Sequential(
            conv(mid_channels, out_channels, stride=1),
            drop(),
            norm(out_channels),
            act(),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = partialize(conv)
        conv2 = partialize(conv)
        drop = partialize(drop)
        norm = partialize(norm)
        act = partialize(act)

        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm1 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()
        self.norm2 = norm(out_channels)
        self.act = act()

        if math.prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            conv = self.conv1.__class__
            self.shortcut = conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)

        out = out + shortcut

        out = self.act(out)

        return out


class PreActivationBlock(nn.Module):
    """Pre-activation version of BasicBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = partialize(conv)
        conv2 = partialize(conv)
        drop = partialize(drop)
        norm = partialize(norm)
        act = partialize(act)

        self.norm1 = norm(in_channels)
        self.act = act()
        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm2 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()

        if math.prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            conv = self.conv1.__class__
            self.shortcut = conv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = self.norm1(x)
        out = self.act(out)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.drop1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.drop2(out)

        out = out + shortcut
        return out
