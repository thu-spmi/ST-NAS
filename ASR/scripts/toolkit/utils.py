'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
This file includes several functions for the convience of work
'''

import random
import os
import sys
import argparse
import struct
import re
import matplotlib.pyplot as plt
import scipy.signal
import math


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def highlight_msg(msg: str):
    len_str = len(msg)
    if '\n' in msg:
        msg = msg.split('\n')
        msg = '\n'.join(['# ' + line + ' #' for line in msg])
    else:
        msg = '# ' + msg + ' #'
    print('\n' + "#"*(len_str + 4))
    print(msg)
    print("#"*(len_str + 4) + '\n')


def packLog(PATH):
    os.system("rm log.tar.gz")
    os.system("tar -cvf log.tar.gz %s" % PATH)


def logType(logfile):
    logfile = logfile.split('/')[-1]

    if "archweight" in logfile:
        return "arch"
    elif "train" in logfile:
        return "train"
    elif "eval" in logfile:
        return "eval"
    else:
        raise NotImplementedError


def read(bindata, N=1):
    output = struct.unpack("%dd" % N, bindata[:8*N])
    bindata = bindata[8*N:]
    if N == 1:
        return bindata, int(output[0])
    else:
        return bindata, [x for x in output]


def _read_arch(archfile):
    loss = []
    bindata = None
    with open(archfile, "rb") as f_in:
        bindata = f_in.read()

    max_node = 0
    bindata, num_blocks = read(bindata)
    for b in range(num_blocks):
        loss.append([])
        bindata, num_nodes = read(bindata)
        if num_nodes > max_node:
            max_node = num_nodes
        for n in range(num_nodes):
            loss[-1].append([])
            bindata, num_ops = read(bindata)
            bindata, len_names = read(bindata)

            names = struct.unpack("%ds" % len_names, bindata[:len_names])[
                0].decode("utf8")

            bindata = bindata[len_names:]
            loss[-1][-1].append(names)

            bindata, len_probs = read(bindata)
            loss[-1][-1].append(bindata[:len_probs])
            bindata = bindata[len_probs:]
    return loss


def mergeLog(logfiles):
    log_type = logType(logfiles[0])
    for file in logfiles:
        assert logType(file) == log_type

    tmp_file = ''
    if log_type == "arch":
        allarch = _read_arch(logfiles[0])
        for log in logfiles[1:]:
            newarch = _read_arch(log)
            for b, block in enumerate(newarch):
                for n, node in enumerate(block):
                    allarch[b][n][1] += node[1]

        # number of blocks
        bindata = struct.pack('d', len(allarch))
        for i, block in enumerate(allarch):
            # number of nodes in each block
            bindata += struct.pack('d', len(block))
            for j, node in enumerate(block):
                names = node[0]
                num_ops = len(names.split(','))
                # number of candidate operations in each node
                bindata += struct.pack('d', num_ops)
                # length of names of candidate operations
                bindata += struct.pack('d', len(names))
                # names of all candidate operations
                bindata += struct.pack('%ds' %
                                       (len(names)), names.encode("utf8"))
                # length of probs (in bytes)
                bindata += struct.pack('d', len(node[1]))
                # probs of each node
                bindata += node[1]

        tmp_file = str(random.random())[2:] + log_type + '.bin'
        with open(tmp_file, 'wb') as file:
            file.write(bindata)

        return tmp_file
    else:
        tmp_file = str(random.random())[2:] + log_type + '.LOG'

    log = struct.pack('')
    interval = struct.pack('s', '\n'.encode("utf8"))

    for logi in logfiles:
        with open(logi, 'rb') as file:
            log += interval + file.read()
    log = log[1:]

    with open(tmp_file, 'wb') as file:
        file.write(log)

    return tmp_file


def dealLog(args):
    logfiles = args.logfiles
    #print(logfiles)
    if len(logfiles) > 1:
        mergefile = mergeLog(logfiles)
        if args.outfile is not None:
            with open(mergefile, 'rb') as f_in:
                with open(args.outfile, 'wb') as f_out:
                    f_out.write(f_in.read())
        packLog(mergefile)
        os.system("rm %s" % mergefile)
    else:
        packLog(logfiles[0])


def _getlatest(dir, pattern, isckpt=False):
    assert os.path.exists(dir)
    files = os.listdir(dir)

    match_files = []
    for file in files:
        if re.search(r"%s" % pattern, file) is not None:
            match_files.append(file)

    assert len(match_files) != 0
    if isckpt:
        latest = sorted(match_files, key=lambda item: int(
            item.split('.')[-2]))[-1]
    else:
        latest = sorted(match_files)[-1]
    print(latest)
    return dir + '/' + latest


def traverse_file(dirname):
    if dirname[-1] != "/":
        dirname += "/"
    files = os.listdir(dirname)
    for file in files:
        file = dirname + file
        if os.path.isdir(file):
            for item in traverse_file(file):
                yield item
        elif '.DS_Store' not in file:
            yield file


def _plot_loss(file):
    # plt.rcParams['figure.figsize'] = [6.22, 3.5]
    print("Reading: {}".format(file))
    log = []
    with open(file, 'r') as f:
        for line in f:
            if "loss:" in line:
                loss = float(line.split('loss: ')[-1])
                log.append(loss)

    if "train" in file.split('/')[-1]:
        log = scipy.signal.savgol_filter(log, 51, 5)
        plt.semilogy(log)
        plt.title("Loss")
        plt.ylabel("Training Loss")
        plt.xlabel("Steps")
    elif "eval" in file.split('/')[-1]:
        plt.plot(log)
        if len(log) >= 5:
            plt.xticks(list(range(0, len(log), len(log)//5)),
                       list(range(1, len(log)+1, len(log)//5)))
        else:
            plt.xticks(list(range(len(log))), list(range(1, len(log)+1)))
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.title("Validation loss")
    else:
        print("unexpected log name.")
        return
    os.makedirs('figures', exist_ok=True)
    plt.savefig("figures/{}.png".format(file), dpi=300)


def _plot_arch_new(file):
    # plt.rcParams['figure.figsize'] = [10, 3.5]
    print("Reading: {}".format(file))

    loss = []
    bindata = None
    with open(file, "rb") as f_in:
        bindata = f_in.read()

    max_probs = 0.
    min_probs = 1.
    max_node = 0
    plot_interval = 0
    bindata, num_blocks = read(bindata)
    for b in range(num_blocks):
        loss.append([])
        bindata, num_nodes = read(bindata)
        if num_nodes > max_node:
            max_node = num_nodes
        for n in range(num_nodes):
            loss[-1].append([])
            bindata, num_ops = read(bindata)
            bindata, len_names = read(bindata)
            #print(len_names)
            names = struct.unpack("%ds" % len_names, bindata[:len_names])[
                0].decode("utf8")
            #print(names)
            bindata = bindata[len_names:]
            loss[-1][-1].append(names)

            bindata, len_probs = read(bindata)
            num_probs = len_probs//8
            bindata, probs = read(bindata, num_probs)
            max_probs = max(max(probs), max_probs)
            min_probs = min(min(probs), min_probs)
            #print(probs)
            for o in range(num_ops):
                loss[-1][-1].append(probs[o::num_ops])
                if len(loss[-1][-1][-1]) > plot_interval:
                    plot_interval = len(loss[-1][-1][-1])

    plot_interval = int(math.pow(10, math.log10(plot_interval)//1))

    fig, subplot = plt.subplots(max_node, num_blocks, figsize=[
                                2*num_blocks+0.2, 0.5+2*max_node])
    if max_node == 1:
        subplot = [subplot]

    plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)    # fontsize of the tick labels

    legends = []
    for b, block in enumerate(loss):
        for n, node in enumerate(block):
            legends.append(node[0].split(','))
            subplot[n][b].set_ylim(min_probs-0.02, max_probs+0.02)
            subplot[n][b].set_xticks(
                list(range(0, len(node[1]), plot_interval)))
            subplot[n][b].set_xticklabels(
                [x//plot_interval for x in list(range(0, len(node[1]), plot_interval))])
            for o, op in enumerate(node[1:]):
                subplot[n][b].plot(op)

    for row in range(max_node):
        for col in range(num_blocks):
            subplot[row][col].grid(ls='--')
            subplot[row][col].set_box_aspect(1)
            if col == 0:
                subplot[row][col].legend(
                    legends[row], loc='upper left', fontsize=8)
            else:
                subplot[row][col].set_yticklabels([])

    subplot[0][2].text(0, -0.2, " "*10)
    pw = "{:.0f}".format(math.log10(plot_interval))
    fig.text(0.5, 0.04, r"Step ($\times 10^{%s})$" % pw, ha='center')

    subplot[0][0].set_ylabel("Architecture Probability")
    namestr = file.split('/')[-1]
    namestr = '.'.join(namestr.split('.')[:-1])
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig("figures/{}.png".format(namestr), dpi=300)


def plot(args):
    tmp_logfile = None
    if len(args.logfile) > 1:
        tmp_logfile = mergeLog(args.logfile)
    else:
        tmp_logfile = args.logfile[0]
    if tmp_logfile.split('.')[-1] == "LOG":
        _plot_loss(tmp_logfile)
    elif tmp_logfile.split('.')[-1] == "bin":
        _plot_arch_new(tmp_logfile)
    else:
        raise NotImplementedError
    if len(args.logfile) > 1:
        os.system("rm {}".format(tmp_logfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()
    cat_parser = subparser.add_parser("concat")
    cat_parser.add_argument("-i", type=str, nargs="+",
                            action="store", dest="logfiles")
    cat_parser.add_argument("-o", type=str, action="store", dest="outfile")
    cat_parser.set_defaults(func=dealLog)

    plot_parser = subparser.add_parser("plot")
    plot_parser.add_argument("-i", type=str, nargs='+',
                             default=None, dest="logfile")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
