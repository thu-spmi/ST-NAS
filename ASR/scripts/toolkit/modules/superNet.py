'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
This is the top interface of NAS model
'''

import torch
import torch.nn as nn
from .Block import *


def _graph_add(graphinfo, i):
    outgraph = []
    for tail, head, op in graphinfo:
        outgraph.append((tail+i, head+i, op))
    return outgraph


def _get_module_name(module: nn.Module) -> str:
    return str(module).split('(')[0]


def _fetch_minmax(graphinfo) -> int:
    return min([tail for (tail, _, _) in graphinfo]), max([head for (_, head, _) in graphinfo])


def _generate_odims(num_blocks, odims):
    if isinstance(odims, int):
        out_chs = [odims for _ in range(num_blocks)]
    elif len(odims) < num_blocks:
        divied = num_blocks//len(odims)
        tmp_chs = odims
        out_chs = []
        for oc in tmp_chs[:-1]:
            out_chs += [oc]*divied
        out_chs += [tmp_chs[-1]]*(num_blocks-len(out_chs))
    else:
        out_chs = odims
    return out_chs


class MySequential(nn.Sequential):
    def __init__(self):
        super(MySequential, self).__init__()

    def insert(self, name, pos, module):
        if name in self._modules:
            raise KeyError("Key already exists")
        if pos < 0 or pos > len(self._modules):
            raise IndexError("Index out of range")

        keys = list(self._modules.keys())[pos:]
        self.add_module(name, module)
        for k in keys:
            self._modules.move_to_end(k)

    def forward(self, features, lens):
        for i, module in enumerate(self):
            try:
                features, lens = module(features=features, lens=lens)
            except:
                features = module(features)
        return features, lens


class superNet(nn.Module):
    def __init__(self, obj=None):
        super(superNet, self).__init__()
        if obj is None:
            self.num_blocks = 0
            self.layers = MySequential()
            self.blocks = []
            self.fixed_part = []
        else:
            self.num_blocks = obj.num_blocks
            self.layers = obj.layers
            self.blocks = obj.blocks
            self.fixed_part = obj.fixed_part

    def add_block(self, name, block):
        self.num_blocks += 1
        self.layers.add_module(name, block)
        self.blocks.append(block)

    def add_fixed(self, name, fixed_layer, pos=None):
        if pos is None:
            self.layers.add_module(name, fixed_layer)
        else:
            self.layers.insert(name, pos, fixed_layer)
        self.fixed_part.append(fixed_layer)

    def printGraph(self):
        net_graph = []
        tail = 0
        for module in self.layers:
            if isinstance(module, ParentBlock):
                block_graph = module.printGraph()
                min_node, max_node = _fetch_minmax(block_graph)
                if min_node == tail:
                    tail = max_node
                else:
                    block_graph = _graph_add(block_graph, tail)
                    tail = max_node + tail
                net_graph += block_graph
            else:
                net_graph.append((tail, tail+1, _get_module_name(module)))
                tail += 1
        return net_graph

    def forward(self, features, lens):
        output, lens_o = self.layers(features, lens)
        return output, lens_o

    def raiseFixed(self):
        output = []
        for block in self.blocks:
            output += block.raiseFixed()
        output += self.fixed_part
        return output

    def raiseNode(self):
        output = []
        for block in self.blocks:
            output = output + block.raiseNode()
        return output

    @property
    def num_block(self):
        return self.num_blocks

    def __getitem__(self, index):
        return self.blocks[index]

    def _fetch_best(self):
        output = []
        for block in self.blocks:
            output.append(block._fetch_best())
        return output


class repeatNet(superNet):
    def __init__(self, BLOCK: ParentBlock, config: list, block_list: list, idim: int, odims, num_classes: int, dropout: float = None, args_block: tuple = None):
        r'''
        base supetnet for repeatly stacking the same kind of Block
        '''
        super(repeatNet, self).__init__()
        num_blocks = block_list[0]
        assert len(config) == num_blocks, "Length of configuration:{} mismatch number of blocks:{}".format(
            len(config), num_blocks)

        odims = _generate_odims(num_blocks, odims)
        odims = [idim] + odims

        args = (dropout,) + args_block if args_block is not None else (dropout, )
        for i in range(num_blocks):
            self.add_block("block%d" % i, BLOCK(
                config[i], odims[i], odims[i+1], *args))

        self.add_fixed("linear1", nn.Linear(odims[-1], num_classes))


class UniNodeNet(repeatNet):
    def __init__(self, **kwargs):
        super(UniNodeNet, self).__init__(WrapNode, **kwargs)

    @staticmethod
    def len_config():
        r'Return length of config in each block.'
        return [1]


class BiResidualNet(repeatNet):
    def __init__(self, **kwargs):
        super(BiResidualNet, self).__init__(
            ResidualBlock, **kwargs, args_block=(BasicBlock,))

    @staticmethod
    def len_config():
        r'Return length of config in each block.'
        return [3]


class ResidualNet(repeatNet):
    def __init__(self, **kwargs):

        super(ResidualNet, self).__init__(
            ResidualBlock, **kwargs, args_block=(WrapNode,))

    @staticmethod
    def len_config():
        r'Return length of config in each block.'
        return [2]


class BiNodeNet(repeatNet):
    def __init__(self, **kwargs):
        super(BiNodeNet, self).__init__(BasicBlock, **kwargs)

    @staticmethod
    def len_config():
        r'Return length of config in each block.'
        return [2]


class SubLayer(nn.Module):
    def __init__(self, coeff: int):
        super(SubLayer, self).__init__()
        self.step = coeff

    def forward(self, features, lens):
        return features[::, ::self.step, ::], lens//self.step


class MFRNet(UniNodeNet):
    def __init__(self, pos_subsample: int, idim_sub:int, odim_sub:int, dropout_sub:float, **kwargs) -> None:
        super(MFRNet, self).__init__(**kwargs)
        sublayer = op_init("tdnn-C1S3D1Nln", idim_sub, odim_sub, dropout_sub)
        self.add_fixed("sublayer", sublayer, pos_subsample)


NetType = {
    "UniNodeNet": UniNodeNet,
    "BiNodeNet": BiNodeNet,
    "ResidualNet": ResidualNet,
    "BiResidualNet": BiResidualNet,
    "MFRNet": MFRNet
}
