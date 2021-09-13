import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
import numpy as np


class WaveNetModel(BaseModel):
    def __init__(self,
                 n_layers=10,
                 n_blocks=4,
                 n_dilation_channels=32,
                 n_residual_channels=32,
                 n_skip_channels=256,
                 n_end_channels=256,
                 n_classes=256,
                 output_length=32,
                 kernel_size=2,
                 global_conditioning=None,
                 bias=False):
        super().__init__()

        # Parameters
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.n_dilation_channels = n_dilation_channels
        self.n_residual_channels = n_residual_channels
        self.n_skip_channels = n_skip_channels
        self.n_end_channels = n_end_channels
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.output_length = output_length
        self.global_conditioning = global_conditioning

        # build model
        self.receptive_field = self.calc_receptive_fields(self.n_layers, self.n_blocks)

        # 1x1 convolution to create channels
        self.causal = nn.Conv1d(self.n_classes, self.n_residual_channels, kernel_size=1, bias=bias)

        # Residual block
        self.res_blocks = ResidualStack(self.n_layers,
                                        self.n_blocks,
                                        self.n_residual_channels,
                                        self.n_dilation_channels,
                                        self.n_skip_channels,
                                        self.kernel_size,
                                        self.global_conditioning,
                                        bias)

        self.end_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(self.n_skip_channels, self.n_end_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.n_end_channels, self.n_classes, kernel_size=1, bias=True)
        )

        self.output_length = output_length

    def forward(self, input, global_condition=None):
        x = self.causal(input)
        skip_connections = self.res_blocks(x, self.output_length, global_condition)
        output = torch.sum(skip_connections, dim=0)
        output = self.end_net(output)
        output = self.regression(output.view(output.size(0), -1)).view(output.size(0), 2, -1)
        return output

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        num_receptive_fields = 2 ** layer_size * stack_size

        return int(num_receptive_fields)


class ResidualStack(BaseModel):
    def __init__(self, layer_size, stack_size, res_channels, dil_channels, skip_channels, kernel_size=2,
                 global_conditioning=None, bias=False,
                 device=None):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :return:
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size
        self.device = device

        self.res_blocks = nn.ModuleList(
            self.stack_res_block(res_channels, dil_channels, skip_channels, kernel_size, global_conditioning, bias))

    def build_dilations(self):
        dilations = []

        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, res_channels, dil_channels, skip_channels, kernel_size, global_conditioning, bias):
        """
        Prepare dilated convolution blocks by layer and stack size
        """
        res_blocks = []
        dilations = self.build_dilations()

        for d in dilations:
            res_blocks.append(ResidualBlock(res_channels,
                                            dil_channels,
                                            skip_channels,
                                            d,
                                            kernel_size=kernel_size,
                                            global_conditioning=global_conditioning,
                                            bias=bias))

        return res_blocks

    def forward(self, x, skip_size, global_condition=None):
        """
        :param x: Input for the operation
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for i in range(len(self.build_dilations())):
            output, skip = self.res_blocks[i](output, skip_size, global_condition)
            skip_connections.append(skip)

        return torch.stack(skip_connections)

    def to_device(self, device):

        for i, _ in enumerate(self.res_blocks):
            self.res_blocks[i].to(device)

        # if torch.cuda.device_count() > 1:
        #    block = torch.nn.DataParallel(block)


class ResidualBlock(BaseModel):

    def __init__(self, res_channels: int, dil_channels, skip_channels, dilation, kernel_size, global_conditioning=None,
                 bias=False):
        """
        Thanks to https://github.com/golbin/WaveNet

        :param res_channels: number of residual channels
        :param skip_channels: number of skip channels
        :param dilation: dilation size
        :param kernel_size: kernel size
        :param bias: is there the bias?
        """
        super(ResidualBlock, self).__init__()

        self.global_conditioning = global_conditioning

        self.dilated_tanh = DilatedCausalConv1d(res_channels, dil_channels, dilation=dilation, kernel_size=kernel_size,
                                                bias=bias)
        self.dilated_sigmoid = DilatedCausalConv1d(res_channels, dil_channels, dilation=dilation,
                                                   kernel_size=kernel_size,
                                                   bias=bias)
        self.conv_res = torch.nn.Conv1d(dil_channels, res_channels, 1, bias=bias)
        self.conv_skip = torch.nn.Conv1d(dil_channels, skip_channels, 1, bias=bias)

        self.globalc_weight = None
        if self.global_conditioning:
            self.globalc_weight_tanh = nn.Linear(self.global_conditioning[0], self.global_conditioning[1], bias=False)
            self.globalc_weight_sigmoid = nn.Linear(self.global_conditioning[0], self.global_conditioning[1],
                                                    bias=False)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size, global_condition=None):
        """
        :param x: Input of the residual block
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output_tanh = self.dilated_tanh(x)
        output_sigmoid = self.dilated_sigmoid(x)
        if self.global_conditioning:
            output_tanh = output_tanh + self.globalc_weight_tanh(global_condition)
            output_sigmoid = output_sigmoid + self.globalc_weight_sigmoid(global_condition)

        gated_tanh = self.gate_tanh(output_tanh)
        gated_sigmoid = self.gate_sigmoid(output_sigmoid)
        gated = gated_tanh * gated_sigmoid

        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output = output + input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]

        return output, skip


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet """

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, bias=False):
        """
        Thanks to https://github.com/golbin/WaveNet

        :param channels: number of channels for the CausalConv
        :param kernel_size: kernel size
        :param dilation: dilation size
        :param bias: Is there a bias?
        """
        super(DilatedCausalConv1d, self).__init__()

        self.padding = (kernel_size - 1) * dilation

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=self.padding,
                                    bias=bias)

    def forward(self, x):
        output = self.conv(x)

        if self.padding == 0:
            return output

        return output[:, :, :-self.conv.padding[0]]