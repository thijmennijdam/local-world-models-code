"""PointNet encoder for XYZ point clouds.

Vendored from 3D Diffusion Policy (Ze et al., 2024; MIT licence, see
THIRD_PARTY_NOTICES.md), file
``diffusion_policy_3d/model/vision/pointnet_extractor_affordance.py``.
Only ``PointNetEncoderXYZ`` is kept; the code is unchanged.
"""

import torch
import torch.nn as nn
from termcolor import cprint


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        block_channels: list = None,
        debug: bool = False,
        **kwargs,
    ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            out_channels (int): output dimension after final projection
            use_layernorm (bool): whether to use layer normalization
            final_norm (str): normalization type for final projection ("layernorm" or "none")
            use_projection (bool): whether to use final projection
            block_channels (list): MLP hidden layer dimensions (default: [64, 128, 256])
            debug (bool): whether to print debug messages
        """
        super().__init__()
        block_channel = block_channels if block_channels is not None else [64, 128, 256]
        self.block_channel = block_channel
        self.debug = debug

        assert in_channels == 3, cprint(
            f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red"
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        self.out_channels = out_channels

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            if self.debug:
                cprint("[PointNetEncoderXYZ] not use projection", "yellow")

        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)

    def forward(self, x):
        if self.debug:
            cprint(f"[PointNetEncoderXYZ - forward] x.shape: {x.shape}", "cyan")
        x = self.mlp(x)
        if self.debug:
            cprint(
                f"[PointNetEncoderXYZ - forward] x.shape after mlp: {x.shape}", "cyan"
            )
        x = torch.max(x, 1)[0]
        if self.debug:
            cprint(
                f"[PointNetEncoderXYZ - forward] x.shape after max: {x.shape}", "cyan"
            )
        x = self.final_projection(x)
        if self.debug:
            cprint(
                f"[PointNetEncoderXYZ - forward] x.shape after final projection: {x.shape}",
                "cyan",
            )
        return x

    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()

    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    def output_shape(self):
        return self.out_channels
