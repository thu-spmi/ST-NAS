'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
Intermediate structure of NAS model.
Block is used to warpped up node(s)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Node import NODE
from .Edge import op_init, Zero


def NODEorOP(config: list, idim: int, odim: int, dropout=None) -> nn.Module:
    assert len(config) > 0, "Invalid candidate number: 0."
    if len(config) == 1:
        return op_init(config[0], idim, odim, dropout)
    else:
        return NODE(config, idim, odim, dropout)


def _select_max(node: NODE):
    _, idx_max = torch.max(node.raiseArchProbs(), dim=0)
    return _get_module_name(node.candidates[idx_max])


def _get_module_name(module: nn.Module) -> str:
    return str(module).split('(')[0]


def _fetch_minmax(graphinfo) -> int:
    return min([tail for (tail, _, _) in graphinfo]), max([head for (_, head, _) in graphinfo])


class ParentBlock(nn.Module):
    def __init__(self):
        super(ParentBlock, self).__init__()
        self.num_nodes = 0
        self.nodes = nn.ModuleList()

    @property
    def num_node(self):
        return self.num_nodes

    def add_node(self, node):
        self.num_nodes += 1
        self.nodes.append(node)

    def __getitem__(self, index):
        return self.nodes[index]

    def raiseNode(self):
        return [x for x in self.nodes]

    def raiseFixed(self):
        return []

    def specify(self):
        assert self.num_nodes > 0
        if isinstance(self.nodes[0], NODE):
            names = [_select_max(node) for node in self.nodes]
        else:
            names = [_get_module_name(module) for module in self.nodes]
        return names

    def printGraph(self):
        pass

    def _fetch_best(self):
        out = []
        for node in self.nodes:
            out.append(node._fetch_best())
        return out


class WrapNode(ParentBlock):
    def __init__(self, config: list, idim: int, odim: int, dropout=None):
        super(WrapNode, self).__init__()
        assert len(config) == 1, "Length of configuration candidates:{} mismatchs number of nodes:{} in WrapNode".format(
            len(config), 1)

        self.add_node(NODEorOP(config[0], idim, odim, dropout))

    def forward(self, features: torch.Tensor, lens):
        output, lens_o = self.nodes[0].forward(features=features, lens=lens)
        return output, lens_o

    def printGraph(self):
        names = self.specify()
        return [(0, 1, names[0])]

    @staticmethod
    def len_config():
        return 1


class BasicBlock(ParentBlock):
    def __init__(self, config: list, idim: int, odim: int, dropout=None):
        super(BasicBlock, self).__init__()
        assert len(config) == 2, "Length of configuration candidates:{} mismatchs number of nodes:{} in Block".format(
            len(config), 2)

        assert odim % 2 == 0
        self.add_node(NODEorOP(config[0], idim, odim//2, dropout))
        self.add_node(NODEorOP(config[1], idim, odim//2, dropout))

    def forward(self, features: torch.Tensor, lens):
        orderNode = self.nodes[0]
        reverseNode = self.nodes[1]
        orderOut, order_lens = orderNode(features=features, lens=lens)
        reverseOut, reverse_lens = reverseNode(
            features=features.flip(1), lens=lens)
        if orderOut.size() != reverseOut.size():
            print("{}.{}.{}".format(features.size(),
                                    orderOut.size(), reverseOut.size()))
            print(
                self.orderNode.candidate_OPs[self.orderNode.actived_index[0]])
            print(
                self.reverseNode.candidate_OPs[self.reverseNode.actived_index[0]])
        assert orderOut.size() == reverseOut.size(
        ), ("Tensors of order output and reverse output give different sizes. {}.{}.{}".format(features.size(), orderOut.size(), reverseOut.size()))
        flipreverse = reverseOut.flip(1)
        output = torch.cat((orderOut, flipreverse), 2)

        return output, order_lens

    def printGraph(self):
        names = self.specify()
        graphinfo = []
        return [(0, 1, names[0]), (0, 1, names[1]+"reverse")]

    @staticmethod
    def len_config():
        return 2


class ResidualBlock(ParentBlock):
    def __init__(self, config: list, idim: int, odim: int, dropout=None, BaseBlock: ParentBlock = None):
        super(ResidualBlock, self).__init__()
        self.base = BaseBlock(config[:-1], idim, odim, dropout)
        for node in self.base.nodes:
            self.add_node(node)

        if len(config[-1]) == 1:
            # build the derived model
            self.add_node(NODEorOP(config[-1], idim, odim, dropout))
        else:
            if idim == odim:
                self.add_node(
                    NODEorOP(['zero', 'identity'], idim, odim, dropout))
            else:
                self.add_node(NODEorOP(['zero', 'lstm'], idim, odim, dropout))

    def forward(self, features: torch.Tensor, lens):
        def _iszero(module) -> bool:
            return isinstance(module, Zero)

        x, lens_o = self.base(features=features, lens=lens)

        if _iszero(self.nodes[-1]):
            return x, lens_o
        else:
            shortcut, lens_short = self.nodes[-1](features=features, lens=lens)

            output = x + shortcut
            return output, lens_o

    def printGraph(self):
        names = self.specify()
        graphinfo = self.base.printGraph()

        min_node, max_node = _fetch_minmax(graphinfo)
        graphinfo.append((min_node, max_node, names[-1]))
        return graphinfo
