'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
Node is the basic unit of NAS
NAS model constructed through: Node->Block->superNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Edge import *


class NODE(nn.Module):
    def __init__(self, ops: list, dim_in, dim_out, dropout=None):
        super(NODE, self).__init__()
        # "search": arch search
        # "warmup" : involves single op
        # "eval": behave the same as warmup
        if dim_in != dim_out:
            tmp_ops = []
            for op in ops:
                if check_hold_dim(op):
                    print("OP:{} is unable to transform features dimension from {} to {}.".format(
                        op, dim_in, dim_out))
                    print("Automatically reduce OP:{}.".format(op))
                    continue
                tmp_ops.append(op)
            ops = tmp_ops
        self.MODE = "warmup"
        self.num_ops = len(ops)
        self.idx_involved = []
        self.idx_activated = []
        assert self.num_ops > 1, "For none variable node, set as `fixed_part` instead."

        # candidate operations
        self.candidates = nn.ModuleList()
        for OP in ops:
            assert isinstance(OP, str)
            self.candidates.append(
                op_init(OP, dim_in, dim_out, dropout=dropout))

    def _fetch_best(self):
        archs = self.raiseArchProbs()
        _, idx = torch.max(archs, 0)
        return [self.candidates[idx]._get_name()]

    def raiseArchProbs(self, idxList=None, requires_grad=False) -> torch.Tensor:
        def _raiseArchWeights(idxList, requires_grad):
            r'return the probs of specified indices ops,\
            if `idxList`=`None`, return probs of all ops'
            if idxList is None:
                idxList = list(range(self.num_ops))
            if requires_grad:
                arch_weight = []
                for i in idxList:
                    arch_weight.append(self.candidates[i].arch_weight)
                arch_weight = torch.cat(arch_weight, dim=0)
            else:
                arch_weight = torch.zeros(len(idxList))
                with torch.no_grad():
                    for m, i in enumerate(idxList):
                        arch_weight[m] = self.candidates[i].arch_weight.data

            return arch_weight

        probs = F.softmax(_raiseArchWeights(idxList, requires_grad), dim=0)

        return probs

    def modeSwitch(self, new_mode: str):
        if new_mode not in ("search", "warmup", "eval"):
            raise NotImplementedError
        else:
            self.MODE = new_mode

    def forward(self, features, lens):

        idx_activated = self.idx_activated[0]
        idx_involved = self.idx_involved

        if self.MODE in ("warmup", "eval"):
            # involves single candidate OP
            return self.candidates[idx_activated](features=features, lens=lens)
        elif self.MODE == "search":
            prob_sliced = self.raiseArchProbs(
                idxList=idx_involved, requires_grad=True)

            output = 0.
            lens_o = None
            for idx_sliced, idx in enumerate(idx_involved):
                _out, lens_o = self.candidates[idx](
                    features=features, lens=lens)
                if idx == idx_activated:
                    output += (1. - prob_sliced[idx_sliced].detach() +
                               prob_sliced[idx_sliced]) * _out
                else:
                    output += (prob_sliced[idx_sliced] -
                               prob_sliced[idx_sliced].detach()) * _out.detach()
            return output, lens_o

        else:
            raise NotImplementedError
