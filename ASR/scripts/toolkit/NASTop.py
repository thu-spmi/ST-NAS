'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
Top interface of NAS.
'''

import sys 
sys.path.append("./ctc-crf")
from ctc_crf import CTC_CRF_LOSS, WARP_CTC_LOSS
from .modules.Node import NODE
from .modules.superNet import NetType, superNet
from .utils import highlight_msg, count_parameters
import torch
import torch.nn as nn
import timeit
import numpy as np
import datetime
import json
import struct
import torch.nn.functional as F
from itertools import cycle
from collections import OrderedDict


class NasManager(object):
    r'Manager to monitor and control the whole procedure of NAS.'

    def __init__(self, model: nn.DataParallel, DEBUG=False):
        super(NasManager, self).__init__()
        assert hasattr(model.module, 'raiseFixed')
        assert hasattr(model.module, 'raiseNode')

        self.net = model
        self.supernet = self.net.module.infer
        self.DEBUG = DEBUG
        self.num_nodes = 0
        self.nodes = model.module.raiseNode()

        for node in self.nodes:
            self.num_nodes += 1
            assert isinstance(
                node, NODE), "Unexpected type in NodeList of NasManager."
        self.MODE = None

        log_arch = []
        for block in self.supernet.blocks:
            log_arch.append([])
            for _ in block.nodes:
                log_arch[-1].append(struct.pack(''))

        self.log = {'trainset': [], 'evalset': [],
                    "archweight": log_arch}
        self.net_optimizer = None
        self.arch_optimizer = None
        self.net_lr_scheduler = None
        self.arch_lr_scheduler = None

        if self.DEBUG:
            highlight_msg(
                "NasManager is configured not to save the checkpoint.")

    def buildOptim(self, lrNet, lrArch):
        r'To build optimizers for network parameters and architecture parameters.'

        archList = []
        for node in self.nodes:
            for i, OP in enumerate(node.candidates):
                archDict = {}
                archDict['lr'] = lrArch
                archDict['params'] = [OP.arch_weight]
                archList.append(archDict)

        self.arch_optimizer = torch.optim.Adam(archList)

        netList = []
        for node in self.nodes:
            for i, OP in enumerate(node.candidates):
                netDict = {}
                netDict['lr'] = lrNet
                netDict['params'] = []
                for name, params in OP.named_parameters():
                    if "arch_weight" not in name:
                        netDict['params'].append(params)
                netList.append(netDict)

        fixed_part = self.net.module.infer.raiseFixed()
        if fixed_part == []:
            self.net_optimizer = torch.optim.Adam(netList, lr=lrNet)
        else:
            fixed_list = [{'params': layer.parameters()}
                          for layer in fixed_part]
            self.net_optimizer = torch.optim.Adam(fixed_list+netList, lr=lrNet)

    def warmUp(self, trainloader, testloader, args):
        r'Warmup step of NAS'

        assert self.net_optimizer is not None and self.net_lr_scheduler is not None, "Please run build_manager() before calling warmup()."

        if args.resume is None:
            setattr(self.net_lr_scheduler, "e_min", 10)
            setattr(self.net_lr_scheduler, "lr_stop",
                    self.net_optimizer.param_groups[0]['lr'])
            setattr(self.net_lr_scheduler, "len_ahead", 3)
            setattr(self.net_lr_scheduler,
                    "min_loss_improve", args.minLossImprove)

        self.warmupSet()

        # Look forward several steps, if loss doesn't decrease, stop warmup.
        e = self.net_lr_scheduler.e_now
        while True:
            prev_t = timeit.default_timer()
            e += 1
            running_loss = 0.
            for i, minibatch in enumerate(trainloader):
                logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch

                self.net_optimizer.zero_grad()
                # resample a new structure
                self.setManager()
                loss = self.net(logits.cuda(), labels_padded.cuda(
                ), input_lengths.cuda(), label_lengths.cuda())

                with torch.no_grad():
                    partial_loss = torch.mean(loss.cpu())
                    weight = torch.mean(path_weights)
                    running_loss += partial_loss - weight

                loss.backward(loss.new_ones(len(loss)))

                self.net_optimizer.step()
                self.net_optimizer.zero_grad()

                if i % 10 == 9 or args.debug:
                    cur_t = timeit.default_timer()
                    self.logLoss("time: {}, loss: {}".format(
                        cur_t - prev_t, running_loss/(i+1)))
                    print("\repoch: {} | step: {} | time: {:.2f} | loss: {:.2f}".format(
                        e, i+1, cur_t - prev_t, running_loss/(i+1)), end='')
                    prev_t = cur_t

            self.eval()
            test_loss = validate(self, testloader)
            print(" | test loss: {:.2f}".format(test_loss))
            state = self.net_lr_scheduler.step(e, test_loss)

            self.warmupSet()

            self.logLoss("loss: {}".format(test_loss), loc="evalset")
            self.logDump(args.warmupDir, "warmup.epoch{:02d}".format(e))

            if state == 0:
                self.saveModel(
                    "model.warmup.epoch.{}".format(e), args.warmupDir)
            elif state == 1:
                self.saveModel(
                    "model.warmup.epoch.{}".format(e), args.warmupDir)
                self.saveModel("model.bestforas", args.warmupDir)
            elif state == 2:
                break
            torch.cuda.empty_cache()

    def archSearch(self, trainloader, testloader, args):
        r'Arch search step of NAS.'

        assert self.net_optimizer is not None and self.net_lr_scheduler is not None, "Please run build_manager() before calling archsearch()."
        if args.resume is None or self.arch_lr_scheduler is None:
            print("Continue with warmup.")
            # directly call archsearch or resume from warmup, so there is no arch_lr_scheduler
            setattr(self.net_lr_scheduler, "e_min", 0)
            setattr(self.net_lr_scheduler, "e_now", 0)
            setattr(self.net_lr_scheduler, "lr_stop",
                    self.net_optimizer.param_groups[0]['lr']*0.1)
            setattr(self.net_lr_scheduler, "len_ahead", 3)
            setattr(self.net_lr_scheduler, "min_loss_improve", args.minLossImprove)
            self.arch_lr_scheduler = Lr_Scheduler(self.arch_optimizer)
        else:
            print("Resumed from paused arch search.")

       # empty log
        self.log['trainset'] = []
        self.log['evalset'] = []

        netTrloader = trainloader
        archIter = cycle(testloader)
        # config NASManager in arch search mode
        self.archsearchSet()
        e = self.net_lr_scheduler.e_now
        interVal = 1000     # interval for monitoring

        while True:
            e += 1
            running_loss = 0.
            prev_t = timeit.default_timer()
            for i, minibatch in enumerate(netTrloader):
                if i % interVal == 0 and not self.DEBUG and i > 0:
                    self.displayArch(toFile=args.disptofile)
                logits, input_lengths, labels_padded, label_lengths, path_weights = next(
                        archIter)

                self.arch_optimizer.zero_grad()
                self.modeswitch("search")
                self.setManager(setforArch=True)
                arch_loss = self.net(logits, labels_padded, input_lengths, label_lengths)
                arch_loss.backward(arch_loss.new_ones(len(arch_loss)))
                self.arch_optimizer.step()
                self.logLoss(loc="archweight")

                logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
                self.net_optimizer.zero_grad()
                self.modeswitch("eval")
                self.setManager()

                loss = self.net(logits.cuda(), labels_padded.cuda(),
                                input_lengths.cuda(), label_lengths.cuda())
                loss.backward(loss.new_ones(len(loss)))

                with torch.no_grad():
                    partial_loss = torch.mean(loss.cpu())
                    weight = torch.mean(path_weights)
                    running_loss += partial_loss - weight
                self.net_optimizer.step()
                if i % 10 == 9 or args.debug:
                    cur_t = timeit.default_timer()
                    self.logLoss("time: {}, loss: {}".format(
                        cur_t - prev_t, running_loss/(i+1)))
                    print("\repoch: {} | step: {} | time: {:.2f} | loss: {:.2f}".format(
                        e, i+1, cur_t - prev_t, running_loss/(i+1)), end='')
                    prev_t = cur_t

            # eval
            self.eval()
            test_loss = validate(self, testloader)
            print(" | test loss: {:.2f}".format(test_loss))
            state = self.net_lr_scheduler.step(e, test_loss)
            self.displayArch()
            self.archsearchSet()

            self.logLoss("loss: {}".format(test_loss), loc="evalset")
            self.logDump(args.archsearchDir, "as.epoch{:02d}".format(e))

            if state == 0:
                self.saveModel("model.as.epoch.{}".format(
                    e), args.archsearchDir)
            elif state == 1:
                self.saveModel("model.as.epoch.{}".format(
                    e), args.archsearchDir)
                self.saveModel("model.bestforretrain", args.archsearchDir)
            elif state == 2:
                break
            else:
                print("Unknown state: {}".format(state))
                raise NotImplementedError

    def retrain(self, trainloader, testloader, args):
        r'Retrain procedure of NAS.`*config` refers \
        to those arguments used to build NasModel'

        assert self.arch_lr_scheduler is not None, "retrain() MUST be called after arch search is done!"

        # fetch the best op among candidates
        selected_op = self.supernet._fetch_best()

        # build the compact model
        model = build_model(args, candidates=selected_op)

        lr_stop = 1e-5
        e_min = 10
        del self.net
        self.net = None
        self.supernet = None
        self.nodes = None
        self.net_optimizer = None
        self.arch_optimizer = None
        torch.cuda.empty_cache()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrRetrain)
        lr_scheduler = Lr_Scheduler_MileStone(
            optimizer, e_min, lr_stop=lr_stop, look_fw=1, gamma=0.1)

        if args.resumeRetrain is not None:
            print("Resume model from {}".format(args.resumeRetrain))
            checkpoint = torch.load(args.resumeRetrain)
            model.load_state_dict(checkpoint['model'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            del checkpoint

        print(model.module.printGraph())
        print("Model size:{:.2f}M".format(count_parameters(model)/1e6))

        train(model, trainloader, testloader, optimizer,
              lr_scheduler, args.retrainDir, self.DEBUG)

    def displayArch(self, showall=True, toFile=None):
        r'Show the candidates and corresponding probabilities.\
        `showall=True` shows all candidates\
        `showall=False` only shows the candidate with largest\
        probability in each node.'

        disp = ""
        len_name_max = 0
        archname = []
        archweights = []
        max_node = 0
        for block in self.supernet.blocks:
            archname.append([])
            archweights.append([])
            if block.num_node > max_node:
                max_node = block.num_node
            for node in block.nodes:
                opstr = []
                w_innode = node.raiseArchProbs().tolist()
                for OP in node.candidates:
                    thisop = str(OP)
                    thisop = thisop.split('(')[0]
                    if len(thisop) > len_name_max:
                        len_name_max = len(thisop)
                    opstr.append(thisop)
                archweights[-1].append(w_innode)
                archname[-1].append(opstr)

        if showall:
            # display first node for all blocks
            for o in range(max([len(x[0]) for x in archweights])):
                for b in range(self.supernet.num_block):
                    if o < len(archweights[b][0]):
                        name = (len_name_max -
                                len(archname[b][0][o]))*' ' + archname[b][0][o]
                        disp += "{}:{:6.2f}% ".format(name,
                                                      archweights[b][0][o] * 100)
                    else:
                        disp += " "*(len_name_max+9)
                disp += "\n"

            # display other nodes
            for n in range(1, max_node):
                for _ in range(self.supernet.num_block):
                    disp += "-"*(len_name_max+8) + " "
                disp += "\n"

                len_ops = [0]*self.supernet.num_block
                for b in range(len(archweights)):
                    if n < len(archweights[b]):
                        len_ops[b] = len(archweights[b][n])

                for o in range(max(len_ops)):
                    for b in range(self.supernet.num_block):
                        if n < len(archweights[b]):
                            if o < len(archweights[b][n]):
                                name = (
                                    len_name_max-len(archname[b][n][o]))*' ' + archname[b][n][o]
                                disp += "{}:{:6.2f}% ".format(
                                    name, archweights[b][n][o] * 100)
                            else:
                                disp += " "*(len_name_max+9)
                        else:
                            disp += " "*(len_name_max+9)
                    disp += "\n"

        else:
            # only show the best arch in each candidate
            len_name_max = 0
            for b in range(self.supernet.num_block):
                for n in range(len(archweights[b])):
                    idx_max = archweights[b][n].index(max(archweights[b][n]))
                    archweights[b][n] = archweights[b][n][idx_max]
                    archname[b][n] = archname[b][n][idx_max]
                    if len(archname[b][n]) > len_name_max:
                        len_name_max = len(archname[b][n])

            for b in range(self.supernet.num_block):
                name = (len_name_max-len(archname[b][0]))*' ' + archname[b][0]
                disp += "{}:{:6.2f}% ".format(name, archweights[b][0]*100)
            disp += "\n"

            for n in range(1, max_node):
                for _ in range(self.supernet.num_block):
                    disp += "-"*(len_name_max+8) + " "
                disp += "\n"

                for b in range(self.supernet.num_block):
                    if n < len(archweights[b]):
                        name = (len_name_max -
                                len(archname[b][n]))*' ' + archname[b][n]
                        disp += "{}:{:6.2f}% ".format(name,
                                                      archweights[b][n]*100)
                    else:
                        disp += " "*(len_name_max+9)
                disp += "\n"

        if toFile is None:
            print("\n{}\n".format(disp))
        else:
            with open(toFile, 'w') as file:
                file.write(disp)

    def setManager(self, setforArch=False):
        r'Set the architecture during forward stage.\
        `setforArch=True` when training arch parameters.\
        `setforArch=False` when training network parameters.'

        def _setEdge(node: NODE):
            if node.MODE == "warmup":
                """
                mode:warmup
                sample one op uniformly
                """
                node.idx_activated = [
                    np.random.randint(low=0, high=node.num_ops)]
                node.idx_involved = node.idx_activated

            elif node.MODE == "search":
                """
                mode:search
                """

                node.idx_involved = list(range(node.num_ops))
                probs = node.raiseArchProbs()
                # sample according to probability
                node.idx_activated = [node.idx_involved[probs.multinomial(1)]]
            elif node.MODE == "eval":
                """
                mode:eval
                mode to do evaluation.
                sample one OP from candidates in each NODE according to
                their softmax-probability.
                """
                probs = node.raiseArchProbs()
                node.idx_involved = torch.multinomial(probs, 1).tolist()
                node.idx_activated = node.idx_involved

        """
        Release all gradient of candidates parameters.\
        call this function after parameter updating, or\
        gradients might be lost.
        """
        def _release_grad(node: NODE):
            for OP in node.candidates:
                for params in OP.parameters():
                    params.requires_grad = False
                    del params.grad
                    params.grad = None

        def _build_grad(node: NODE):
            idx_activated = node.idx_activated[0]
            if setforArch:
                for idx in node.idx_involved:
                    node.candidates[idx].arch_weight.requires_grad = True
            else:
                for name, params in node.candidates[idx_activated].named_parameters():
                    if 'arch_weight' not in name:
                        params.requires_grad = True

        # release grad
        self.apply_fn_node(_release_grad)
        self.apply_fn_node(_setEdge)
        # if self.MODE == "eval":
        #     # for eval mode, it is not required to re-build the gradients
        #     return
        # set requires_grad=True for active op
        self.apply_fn_node(_build_grad)

    def modeswitch(self, mode: str):
        r'''
        switch mode of nodes to `mode`
        '''
        assert mode in ("warmup", "eval", "search")
        for node in self.nodes:
            node.MODE = mode

    def saveModel(self, name, PATH):
        r'Save checkpoint.\
        The checkpoint file would be located at `PATH/name.pt`'

        if self.DEBUG:
            print("Debugging, skipped saving model.")
            return
        if PATH[-1] != '/':
            PATH += '/'
        if self.MODE == "warmup":
            torch.save({
                'model': self.net.state_dict(),
                'net_lr_scheduler': self.net_lr_scheduler.state_dict()
            }, PATH+name+'.pt')
        else:
            torch.save({
                'model': self.net.state_dict(),
                'net_lr_scheduler': self.net_lr_scheduler.state_dict(),
                'arch_lr_scheduler': self.arch_lr_scheduler.state_dict()
            }, PATH+name+'.pt')

    def loadModel(self, PATH):
        r'Load checkpoint from `PATH`.'

        checkpoint = torch.load(PATH)
        self.net.load_state_dict(checkpoint['model'])
        self.net_lr_scheduler.load_state_dict(checkpoint['net_lr_scheduler'])

        if 'arch_lr_scheduler' in checkpoint.keys():
            if self.arch_lr_scheduler is None:
                '''
                arch_lr_scheduler is not create when building manager, so there
                might be cases that calling loadModel() before creating arch_lr_scheduler.
                Temporarily save the checkpoint into self.arch_lr_scheduler
                '''
                self.arch_lr_scheduler = Lr_Scheduler(self.arch_optimizer)
                self.arch_lr_scheduler.load_state_dict(
                    checkpoint['arch_lr_scheduler'])
            else:
                self.arch_lr_scheduler.load_state_dict(
                    checkpoint['arch_lr_scheduler'])

    def logLoss(self, loss_msg=None, loc="trainset"):
        assert loc in ("trainset", "evalset", "archweight")

        if loc == "archweight":
            for i, block in enumerate(self.supernet.blocks):
                for j, node in enumerate(block.nodes):
                    archlog = node.raiseArchProbs().tolist()
                    archlog = struct.pack("%dd" % len(archlog), *archlog)
                    self.log['archweight'][i][j] += archlog
        else:
            self.log[loc].append(loss_msg)

    def logDump(self, PATH, nameappd=''):
        if self.DEBUG:
            print("Debugging, skipped log dump.")
            return
        nameappd = datetime.datetime.now().strftime(
            '%Y_%m_%d_%H_%M_%S') if nameappd == '' else nameappd

        if PATH[-1] != '/':
            PATH = PATH + '/'
        for key, value in self.log.items():
            if key in ("archweight") and self.MODE == "warmup":
                continue
            if key == "archweight":
                # number of blocks
                bindata = struct.pack('d', len(value))
                for i, block in enumerate(self.supernet.blocks):
                    # number of nodes in each block
                    bindata += struct.pack('d', block.num_node)
                    for j, node in enumerate(block):
                        # number of candidate operations in each node
                        bindata += struct.pack('d', node.num_ops)
                        # length of names of candidate operations
                        names = ""
                        for op in node.candidates:
                            names += str(op).split('(')[0]
                            names += ','
                        names = names[:-1]
                        bindata += struct.pack('d', len(names))
                        # names of all candidate operations
                        bindata += struct.pack('%ds' %
                                               (len(names)), names.encode("utf8"))
                        # length of probs (in bytes)
                        bindata += struct.pack('d', len(value[i][j]))
                        # probs of each node
                        bindata += value[i][j]

                with open("{}/{}.{}.bin".format(PATH, key, nameappd), 'wb') as file:
                    file.write(bindata)
                continue

            with open("{}/{}.{}.LOG".format(PATH, key, nameappd), 'w+', encoding='utf8') as file:
                file.write('\n'.join(value))

    def archsearchSet(self):
        self.MODE = "search"
        self.modeswitch("search")
        self.net.train()

    def warmupSet(self):
        self.MODE = "warmup"
        self.modeswitch("warmup")
        self.net.train()

    def eval(self):
        self.MODE = "eval"
        self.modeswitch("eval")
        self.net.eval()

    def apply_fn_node(self, fn):
        for node in self.nodes:
            fn(node)


def build_model(args, candidates=None) -> list:
    r'''
    `block_list`:list type, length of num_blocks should match the superNet.lenconfig()
    e.g. `[1,2,3]` denotes there are `1`, `2` and `3` blocks for each type.
    '''
    def _nmcopy(origin: list, lenconfigs, num_blocks) -> list:
        od = []
        assert len(lenconfigs) == len(num_blocks)
        for len_config_block, n_block in zip(lenconfigs, num_blocks):
            od += [[origin.copy() for _ in range(len_config_block)]
                   for _ in range(n_block)]
        return od

    with open(args.nasConfig, 'r') as file:
        data = json.load(file)

    supernet = NetType[data['net']]
    net_configs = data['net_configs']
    block_list = net_configs['block_list']

    if candidates is None:
        candidates = data['candidate']
        if isinstance(candidates, list):
            # the standard arguments
            candidates = _nmcopy(candidates, supernet.len_config(), block_list)
        else:
            # simplified arguments
            base = candidates['base']
            output = _nmcopy(base, supernet.len_config(), block_list)
            for pos, ops in candidates.items():
                if pos == "base":
                    continue
                pos = [int(i) for i in pos.split('-')]
                output[pos[0]][pos[1]] = ops
            candidates = output

    net_configs['config'] = candidates
    if args.MODE == "warmup":
        lossfn = args.lossFnWarmup
    elif args.MODE == "archsearch":
        lossfn = args.lossFnAS
    elif args.MODE == "retrain":
        lossfn = args.lossFnRetrain
    else:
        lossfn = None
        raise ValueError
    model = NasModel(supernet, lossfn, args.lamb, net_configs)
    device = torch.device("cuda")
    model = nn.DataParallel(model)
    model.to(device)
    return model


def build_manager(model, args):
    manager = NasManager(model, args.debug)
    manager.buildOptim(args.lrNetParams, args.lrArchParams)

    # init a net_lr_scheduler, but still have to set attributes(like lr and lr_stop etc.) in specified stages.
    manager.net_lr_scheduler = Lr_Scheduler_MileStone(manager.net_optimizer, 0)
    if args.resume is not None:
        print("Resuming from: {}".format(args.resume))
        manager.loadModel(args.resume)
    return manager


class NasModel(nn.Module):
    def __init__(self, NET: superNet = None, fn_loss='crf', lamb: float = 0.1, net_configs: dict = None):
        super(NasModel, self).__init__()
        if NET is None:
            return

        self.infer = NET(**net_configs)

        if fn_loss == "ctc":
            self.loss_fn = WARP_CTC_LOSS()
        elif fn_loss == "crf":
            self.loss_fn = CTC_CRF_LOSS(lamb=lamb)
        else:
            raise NotImplementedError

    def raiseNode(self):
        return self.infer.raiseNode()

    def raiseFixed(self):
        return self.infer.raiseFixed()

    def printGraph(self):
        supernet_graph = self.infer.printGraph()
        full_graph = supernet_graph
        return ["%d-%d@%s" % (tail, head, op) for (tail, head, op) in full_graph]

    def forward(self, logits, labels_padded, input_lengths, label_lengths):
        # rearrange by input_lengths
        input_lengths, indices = torch.sort(input_lengths, descending=True)
        assert indices.dim() == 1, "input_lengths should have only 1 dim"
        logits = torch.index_select(logits, 0, indices)
        labels_padded = torch.index_select(labels_padded, 0, indices)
        label_lengths = torch.index_select(label_lengths, 0, indices)

        labels_padded = labels_padded.cpu()
        label_lengths = label_lengths.cpu()

        label_list = [labels_padded[i, :x]
                      for i, x in enumerate(label_lengths)]
        labels = torch.cat(label_list)

        netout, lens_o = self.infer(logits, input_lengths)
        netout = F.log_softmax(netout, dim=2)
        loss = self.loss_fn(netout, labels, lens_o.to(torch.int32).cpu(), label_lengths)
        with torch.no_grad():
            if torch.isnan(loss):
                print("nan in loss.")
                sys.exit(-1)
        return loss


def validate(nasmanager: NasManager, testloader) -> float:
    r'Validate the supernet performance on devset. Return `test_loss`'

    sum_test_loss = []
    model = nasmanager.net
    count = 0
    for i, minibatch in enumerate(testloader):
        logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
        nasmanager.setManager()
        if torch.__version__ <= '1.1.0':
            loss = model(logits, labels_padded, input_lengths, label_lengths)
            with torch.no_grad():
                partial_loss = torch.mean(loss.cpu())
        else:
            with torch.no_grad():
                loss = model(logits, labels_padded,
                             input_lengths, label_lengths)
            partial_loss = torch.mean(loss.cpu())

        loss_size = loss.size(0)
        count = count + loss_size
        weight = torch.mean(path_weights)
        real_loss = partial_loss - weight
        real_loss_sum = real_loss * loss_size
        sum_test_loss.append(real_loss_sum.item())

    test_loss = sum(sum_test_loss)/count
    return test_loss


def train(model: nn.Module, trainloader, testloader, optimizer, lr_scheduler, saved_path, debug=True):
    def logDump(losslog, PATH):
        dt = datetime.datetime.now().strftime('%m_%d_%H_%M')
        if PATH[-1] != '/':
            PATH = PATH + '/'

        for key, value in losslog.items():
            with open(PATH+key+dt+'.LOG', 'w+', encoding='utf8') as file:
                file.write('\n'.join(value))

    losslog = {
        "train": [],
        "eval": []
    }
    epoch = lr_scheduler.e_now
    model.train()
    while True:
        # training stage
        epoch += 1
        running_loss = 0.
        prev_t = timeit.default_timer()
        for i, minibatch in enumerate(trainloader):
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            loss = model(logits, labels_padded, input_lengths, label_lengths)

            with torch.no_grad():
                partial_loss = torch.mean(loss.cpu())
                weight = torch.mean(path_weights)
                running_loss += partial_loss - weight

            loss.backward(loss.new_ones(len(loss)))
            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 9 or debug:
                cur_t = timeit.default_timer()
                print("\repoch: {} | step: {} | time: {:.2f} | loss: {:.2f} | lr: {:.0e}".format(
                    epoch, i+1, cur_t - prev_t, running_loss/(i+1), optimizer.param_groups[0]['lr']), end='')

                if not debug:
                    losslog["train"].append("time: {:.2f} | loss: {:.2f}".format(
                        cur_t - prev_t, running_loss/(i+1)))
                prev_t = cur_t

        # cv stage
        model.eval()
        sum_test_loss = []
        count = 0

        for i, minibatch in enumerate(testloader):
            logits, input_lengths, labels_padded, label_lengths, path_weights = minibatch
            if torch.__version__ <= '1.1.0':
                loss = model(logits, labels_padded,
                             input_lengths, label_lengths)
                with torch.no_grad():
                    partial_loss = torch.mean(loss.cpu())
                loss_size = loss.size(0)
                del loss
                loss = None
                torch.cuda.empty_cache()
            else:
                with torch.no_grad():
                    loss = model(logits, labels_padded,
                                 input_lengths, label_lengths)
                partial_loss = torch.mean(loss.cpu())
                loss_size = loss.size(0)
            count = count + loss_size
            weight = torch.mean(path_weights)
            real_loss = partial_loss - weight
            real_loss_sum = real_loss * loss_size
            sum_test_loss.append(real_loss_sum.item())

        test_loss = np.sum(np.asarray(sum_test_loss))/count
        print(" | test loss: {:.2f}{}".format(test_loss, ' '*10))

        state = lr_scheduler.step(epoch, test_loss)

        if debug:
            if state == 2:
                break
            model.train()
            continue
        else:
            losslog["eval"].append("test_loss: {}".format(test_loss))
            logDump(losslog, saved_path)

        if state == 0:
            torch.save({
                'model': model.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, saved_path + "/model.epoch.{:02d}.pt".format(epoch))

        elif state == 1:
            torch.save({
                'model': model.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, saved_path + "/model.epoch.{:02d}.pt".format(epoch))

            # save model for inference
            torch.save(model.module.infer, saved_path +
                       "/model.bestforinfer.pt".format(epoch))
        elif state == 2:
            break
        else:
            print("Unknown state: {}".format(state))
            raise NotImplementedError

        model.train()


class Lr_Scheduler(object):
    def __init__(self, optimizer):
        super(Lr_Scheduler, self).__init__()
        self.optimizer = optimizer

    def state_dict(self):
        output = OrderedDict()
        for name, value in vars(self).items():
            if name == 'optimizer':
                output['optimizer'] = value.state_dict()
                output['lr'] = value.param_groups[0]['lr']
            else:
                output[name] = value
        return output

    def load_state_dict(self, ckpt: OrderedDict):
        def set_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for name in vars(self).keys():
            if name == "optimizer":
                self.optimizer.load_state_dict(ckpt[name])
                set_lr(self.optimizer, ckpt['lr'])
            elif name in ("worse_loss_count", "best_loss", "e_now"):
                setattr(self, name, ckpt[name])


class Lr_Scheduler_MileStone(Lr_Scheduler):
    def __init__(self, optimizer, epoch_min, lr_stop=1e-5, look_fw=1, gamma=0.1, min_loss_improve=0.):
        super(Lr_Scheduler_MileStone, self).__init__(optimizer)
        self.lr_stop = lr_stop
        self.len_ahead = look_fw
        self.worse_loss_count = 0
        self.best_loss = float('inf')
        self.gamma = gamma
        self.e_min = epoch_min
        self.e_now = 0
        self.min_loss_improve = min_loss_improve

    def step(self, global_epoch, loss):
        r'''
        return three state (int)`0`,`1`,`2`.\
        `0`: save ckeckpoint and continue\
        `1`: save checkpoint, inference model and continue\
        `2`: stop training.
        '''
        def adjust_lr(optimizer, gamma):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma

        self.e_now = global_epoch
        if self.e_now <= self.e_min:
            if loss < self.best_loss:
                self.best_loss = loss
            return 0
        elif loss < self.best_loss - self.min_loss_improve:
            self.best_loss = loss
            self.worse_loss_count = 0
            return 1
        else:
            self.worse_loss_count += 1
            if self.worse_loss_count >= self.len_ahead:
                lr = self.optimizer.param_groups[0]['lr']
                print("Validation loss doesn't improve, decay the learning rate from {:.0e} to {:.0e}".format(
                    lr, lr * self.gamma))

                lr = lr * self.gamma
                if lr < self.lr_stop:
                    print("lr: {:.0e} < lr_stop: {:.0e}, terminate training.".format(
                        lr, self.lr_stop))
                    return 2
                else:
                    adjust_lr(self.optimizer, self.gamma)
                    return 0
            else:
                return 0


def _fetch_minmax(graphinfo) -> int:
    return min([tail for (tail, _, _) in graphinfo]), max([head for (_, head, _) in graphinfo])
