import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from .Blocks import DoubleConv, BasicBlock, trunc_normal_init, constant_init, kaiming_init


class MambaLayer(nn.Module):
    """
    Mamba Layer integrating a Selective State-Space Model (SSM).
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension
            d_state=d_state,  # SSM state dimension
            d_conv=d_conv,  # Convolutional dimension
            expand=expand,  # Expansion factor
            # bimamba_type="v3",
            # nframes=num_slices,
        )

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim, "Input channel dimension must match the model dimension."
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        # Reshape and transpose for processing
        x_flat = x.view(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        # Restore original dimensions
        out = x_mamba.transpose(-1, -2).view(B, C, *img_dims)
        return out


class MlpChannel(nn.Module):
    """
    MLP block applied across channel dimensions.
    """

    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, kernel_size=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MBBlock(nn.Module):
    """
    Mamba Block consisting of multiple Mamba layers and an MLP.
    """

    def __init__(self, in_channel, num_slices, depth):
        super(MBBlock, self).__init__()
        self.stage = nn.Sequential(
            *[MambaLayer(dim=in_channel, num_slices=num_slices) for _ in range(depth)]
        )
        self.IN = nn.InstanceNorm3d(in_channel)
        self.mlp = MlpChannel(in_channel, 2 * in_channel)

    def forward(self, x):
        x = self.stage(x)
        x_out = self.IN(x)
        x_out = self.mlp(x_out)
        return x_out


class Mamba_Encoder(nn.Module):
    """
    Mamba Encoder with multiple layers and downsampling.
    """

    def __init__(self, num_channels=[24, 48, 96, 192, 384], layer_nums=[2, 2, 2, 2], in_channels=1,
                 num_slices_list=[64, 32, 16, 8]):
        super(Mamba_Encoder, self).__init__()
        self.channels = num_channels[0]
        self.conv1 = DoubleConv(in_channels, num_channels[0])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Define layers with downsampling
        self.layer1_1 = MBBlock(in_channel=num_channels[0], num_slices=num_slices_list[0], depth=layer_nums[0])
        self.convdown1 = nn.Sequential(
            nn.Conv3d(num_channels[0], num_channels[1], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=num_channels[1]),
            nn.ReLU()
        )
        self.layer2_1 = MBBlock(in_channel=num_channels[1], num_slices=num_slices_list[1], depth=layer_nums[1])
        self.convdown2 = nn.Sequential(
            nn.Conv3d(num_channels[1], num_channels[2], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=num_channels[2]),
            nn.ReLU()
        )
        self.layer3_1 = MBBlock(in_channel=num_channels[2], num_slices=num_slices_list[2], depth=layer_nums[2])
        self.convdown3 = nn.Sequential(
            nn.Conv3d(num_channels[2], num_channels[3], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=num_channels[3]),
            nn.ReLU()
        )
        self.layer4_1 = MBBlock(in_channel=num_channels[3], num_slices=num_slices_list[2], depth=layer_nums[2])
        self.convdown4 = nn.Sequential(
            nn.Conv3d(num_channels[3], num_channels[4], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=num_channels[4]),
            nn.ReLU()
        )

    def _init_weights_kaiming(self, m):
        """
        Initialize weights using Kaiming initialization.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv3d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def init_weight(self):
        """
        Apply weight initialization to all layers.
        """
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.convdown4.apply(self._init_weights_kaiming)

    def forward(self, x):
        """
        Forward pass through the encoder.
        """
        outs = []
        x0 = self.maxpool(self.conv1(x))
        x1 = self.layer1_1(x0)
        x1 = self.convdown1(self.relu(x1))
        x2 = self.convdown2(self.relu(self.layer2_1(x1)))
        x3 = self.convdown3(self.relu(self.layer3_1(x2)))
        x4 = self.convdown4(self.relu(self.layer4_1(x3)))
        outs.extend([x4, x3, x2, x1, x0])
        return outs


if __name__ == '__main__':
    pass
