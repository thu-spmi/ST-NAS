'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
In this file, we implement several OPs for Node
'''

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .maskedbatchnorm1d import MaskedBatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Edge(nn.Module):
    def __init__(self, dropout=None):
        super(Edge, self).__init__()
        self.arch_weight = nn.Parameter(arch_init())

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def _get_name(self) -> str:
        pass


class EncoderLayer(Edge):
    def __init__(self, dim_i, dim_o=None, nhead=8, dim_feedforward=1024, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_i, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_i, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_i)

        self.norm1 = nn.LayerNorm(dim_i)
        self.norm2 = nn.LayerNorm(dim_i)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EncoderLayer, self).__setstate__(state)

    def forward(self, **kwargs):
        src = kwargs["features"]
        src2 = self.self_attn(src, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def _get_name(self) -> str:
        return "transformer"

    def __str__(self) -> str:
        return "Transformer"


class RNNEdge(Edge):
    def __init__(self, dropout) -> None:
        super(RNNEdge, self).__init__(dropout)
        self.rnn = None

    def forward(self, **kwargs):
        x = kwargs['features']
        lens = kwargs['lens']

        self.rnn.flatten_parameters()
        len_total = x.size(1)

        packed_input = pack_padded_sequence(x, lens, batch_first=True)
        packed_output, _ = self.rnn(packed_input)
        rnn_out, lens_out = pad_packed_sequence(
            packed_output, batch_first=True, total_length=len_total)

        if self.dropout is not None:
            rnn_out = self.dropout(rnn_out)
        return rnn_out, lens_out.to(rnn_out.device)


class GRU(RNNEdge):
    def __init__(self, idim, hdim, dropout=None):
        super(GRU, self).__init__(dropout)
        self.rnn = nn.GRU(idim, hdim, batch_first=True)

    def __str__(self):
        return str(self.gru)

    def _get_name(self) -> str:
        return "gru"


class BGRU(RNNEdge):
    def __init__(self, idim, hdim, dropout=None):
        super(BGRU, self).__init__(dropout)
        self.rnn = nn.GRU(idim, hdim, batch_first=True, bidirectional=True)

    def _get_name(self) -> str:
        return "bgru"

    def __str__(self):
        return str(self.bgru)


class LSTM(RNNEdge):
    def __init__(self, idim, hdim, dropout=None):
        super(LSTM, self).__init__(dropout)
        self.rnn = nn.LSTM(idim, hdim, batch_first=True)

    def _get_name(self) -> str:
        return "lstm"

    def __str__(self):
        return "LSTM"


class BLSTM(RNNEdge):
    def __init__(self, idim, hdim, num_layers=1, dropout=None):
        super(BLSTM, self).__init__(dropout)
        if dropout is None:
            dropout = 0.
        self.rnn = nn.LSTM(idim, hdim, num_layers=num_layers,
                           batch_first=True, bidirectional=True)

    def _get_name(self) -> str:
        return "blstm"

    def __str__(self):
        return "BLSTM"


class TDNN(Edge):
    def __init__(self, dim_i, dim_o, half_context=1, stride=1, dilation=1, norm='ln', dropout=None):
        super(TDNN, self).__init__(dropout)
        self.vals = [half_context, stride, dilation, norm]
        padding = half_context * dilation
        self.conv = torch.nn.Conv1d(
            dim_i, dim_o, kernel_size=2*half_context+1, padding=padding, stride=stride, dilation=dilation)

        if norm == "bn":
            self.norm = MaskedBatchNorm1d(dim_o)
        elif norm == "ln":
            self.norm = nn.LayerNorm(dim_o)
        else:
            raise ValueError

    def forward(self, **kwargs):
        features = kwargs['features']
        lens = kwargs['lens']

        tdnn_in = features.transpose(1, 2)
        tdnn_out = self.conv(tdnn_in)
        output = F.relu(tdnn_out)
        output = output.transpose(1, 2)

        if isinstance(self.norm, MaskedBatchNorm1d):
            output = self.norm(output, lens)
        else:
            output = self.norm(output)

        if self.dropout is not None:
            output = self.dropout(output)

        return output, lens*output.size(1)/features.size(1)

    def __str__(self):
        return "TDNN-C{}S{}D{}N{}".format(*(self.vals))

    def _get_name(self):
        return "tdnn-C{}S{}D{}N{}".format(*(self.vals))


class Identity(Edge):
    def __init__(self, **args):
        super(Identity, self).__init__()

    def forward(self, **kwargs):
        features = kwargs['features']
        return features, kwargs['lens']

    def _get_name(self) -> str:
        return "identity"

    def __str__(self):
        return "Identity"


class Zero(Edge):
    def __init__(self, dim_out):
        super(Zero, self).__init__()
        self.out = dim_out

    def forward(self, **kwargs):
        features = kwargs['features']
        return torch.zeros([features.size(0), features.size(1), self.out], dtype=features.dtype, layout=features.layout, device=features.device), kwargs['lens']

    def _get_name(self) -> str:
        return "zero"

    def __str__(self):
        return "Zero"


"""
implementation of initialization of arch parameters
"""


def arch_init():
    #return torch.empty(1).normal_(mean=0,std=0.1)
    return torch.zeros(1)


"""
initialization of different layer
considering leveraging the memory consumption
"""


def op_init(candidate, dim_in, dim_out, dropout=None) -> nn.Module:
    configs = candidate.split('-')
    layer = configs[0]
    if layer == "lstm":
        return LSTM(dim_in, dim_out, dropout=dropout)
    elif layer == "blstm":
        return BLSTM(dim_in, dim_out//2, dropout=dropout)
    elif layer == "gru":
        return GRU(dim_in, dim_out, dropout=dropout)
    elif layer == "bgru":
        return BGRU(dim_in, dim_out//2, dropout=dropout)
    elif layer == "tdnn":
        configs = configs[1]

        def _update(pattern, default):
            r = re.search(pattern, configs)
            if r is not None:
                return type(default)(r.group(0))
            else:
                return default

        half_context = _update(r'(?<=C)\d+', 1)
        stride = _update(r'(?<=S)\d+', 1)
        dilation = _update(r'(?<=D)\d+', 1)
        if dilation > 1:
            print(
                "Dilation is set > 1, which might cause slow computation in some machines.")
        norm = _update(r'(?<=N)(ln|bn)', 'ln')

        return TDNN(dim_in, dim_out, half_context=half_context, stride=stride, dilation=dilation, norm=norm, dropout=dropout)

    elif layer == "identity":
        return Identity()
    elif layer == "transformer":
        return EncoderLayer(dim_in)
    elif layer == "zero":
        return Zero(dim_out)
    else:
        print(candidate)
        raise NotImplementedError


def check_hold_dim(OP: str) -> bool:
    return OP in ["identity", "transformer"]
