## BSD 3-Clause License
##
## Copyright (c) 2021, Andrej Orsula
## All rights reserved.

## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:

## 1. Redistributions of source code must retain the above copyright notice, this
##   list of conditions and the following disclaimer.
##
## 2. Redistributions in binary form must reproduce the above copyright notice,
##   this list of conditions and the following disclaimer in the documentation
##   and/or other materials provided with the distribution.
##
## 3. Neither the name of the copyright holder nor the names of its
##   contributors may be used to endorse or promote products derived from
##   this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
## FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import ocnn
import torch


class OctreeConvRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
        super(OctreeConvRelu, self).__init__()
        self.conv = ocnn.OctreeConv(depth,
                                    channel_in,
                                    channel_out,
                                    kernel_size,
                                    stride)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.relu(out)
        return out


class OctreeConvBnRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1, bn_eps=0.00001, bn_momentum=0.01):
        super(OctreeConvBnRelu, self).__init__()
        self.conv = ocnn.OctreeConv(depth,
                                    channel_in,
                                    channel_out,
                                    kernel_size,
                                    stride)
        self.bn = torch.nn.BatchNorm2d(channel_out,
                                       bn_eps,
                                       bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.bn(out)
        out = self.relu(out)
        return out


class OctreeConvFastRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1):
        super(OctreeConvFastRelu, self).__init__()
        self.conv = ocnn.OctreeConvFast(depth,
                                        channel_in,
                                        channel_out,
                                        kernel_size,
                                        stride)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.relu(out)
        return out


class OctreeConvFastBnRelu(torch.nn.Module):
    def __init__(self, depth, channel_in, channel_out, kernel_size=[3], stride=1, bn_eps=0.00001, bn_momentum=0.01):
        super(OctreeConvFastBnRelu, self).__init__()
        self.conv = ocnn.OctreeConvFast(depth,
                                        channel_in,
                                        channel_out,
                                        kernel_size,
                                        stride)
        self.bn = torch.nn.BatchNorm2d(channel_out,
                                       bn_eps,
                                       bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in, octree):
        out = self.conv(data_in, octree)
        out = self.bn(out)
        out = self.relu(out)
        return out


class OctreeConv1x1Relu(torch.nn.Module):
    def __init__(self, channel_in, channel_out, use_bias=True):
        super(OctreeConv1x1Relu, self).__init__()
        self.conv1x1 = ocnn.OctreeConv1x1(channel_in, channel_out, use_bias)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.conv1x1(data_in)
        out = self.relu(out)
        return out


class OctreeConv1x1BnRelu(torch.nn.Module):
    def __init__(self, channel_in, channel_out, use_bias=True, bn_eps=0.00001, bn_momentum=0.01):
        super(OctreeConv1x1BnRelu, self).__init__()
        self.conv1x1 = ocnn.OctreeConv1x1(channel_in, channel_out, use_bias)
        self.bn = torch.nn.BatchNorm2d(channel_out, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.conv1x1(data_in)
        out = self.bn(out)
        out = self.relu(out)
        return out


class LinearRelu(torch.nn.Module):
    def __init__(self, channel_in, channel_out, use_bias=True):
        super(LinearRelu, self).__init__()
        self.fc = torch.nn.Linear(channel_in, channel_out, use_bias)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.fc(data_in)
        out = self.relu(out)
        return out


class LinearBnRelu(torch.nn.Module):
    def __init__(self, channel_in, channel_out, use_bias=True, bn_eps=0.00001, bn_momentum=0.01):
        super(LinearBnRelu, self).__init__()
        self.fc = torch.nn.Linear(channel_in, channel_out, use_bias)
        self.bn = torch.nn.BatchNorm1d(channel_out, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data_in):
        out = self.fc(data_in)
        out = self.bn(out)
        out = self.relu(out)
        return out
