'''
Copyright 2020 SPMI-CAT
Modified by Zheng Huahuan 2020
this script is for calculating the final results of trained model
'''

import sys
import os
import torch
import numpy as np
import argparse
d = os.path.dirname(__file__)
parent_path = os.path.dirname(d)
sys.path.append(parent_path)
sys.path.append('./ctc-crf')
import kaldi_io
from toolkit.utils import count_parameters

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="inference network")
    parser.add_argument("--input_scp", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_unit", type=int)
    parser.add_argument("--hdim", type=int, default=320)
    parser.add_argument("--dropout", type=int, default=0.5)
    parser.add_argument("--feature_size", type=int, default=120)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--nj", type=int)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    batch_size = 512
    assert args.resume is not None

    print("Load whole model")
    model = torch.load(args.resume)

    print("Model size:{:.2f}M".format(count_parameters(model)/1e6))

    print("Model loaded from checkpoint: {}.".format(args.resume))
    model.eval()
    model.cuda()
    n_jobs = args.nj
    writers = []
    write_mode = 'w'
    if sys.version > '3':
        write_mode = 'wb'

    for i in range(n_jobs):
        writers.append(
            open('{}/decode.{}.ark'.format(args.output_dir, i+1), write_mode))

    with open(args.input_scp) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        utt, feature_path = line.split()
        feature = kaldi_io.read_mat(feature_path)
        input_lengths = torch.IntTensor([feature.shape[0]])
        feature = torch.from_numpy(feature[None])
        feature = feature.cuda()

        netout, _ = model.forward(feature, input_lengths)
        r = netout.cpu().data.numpy()
        r[r == -np.inf] = -1e16
        r = r[0]
        kaldi_io.write_mat(writers[i % n_jobs], r, key=utt)

    for i in range(n_jobs):
        writers[i].close()
