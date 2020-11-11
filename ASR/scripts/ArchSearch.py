'''
Copyright 2020 Tsinghua University
Author: Zheng Huahuan (zhh20@mails.tsinghua.edu.cn)
'''

import os
import argparse
import numpy as np
import torch
#import torch before import ctc_crf_base
import ctc_crf_base
from toolkit.NASTop import build_model, build_manager
from toolkit.dataset_pickle import SpeechDataset, PadCollate
from toolkit.utils import count_parameters, highlight_msg
from torch.utils.data import DataLoader


torch.backends.cudnn.deterministic = True
# This line in rid of some conditional errors.
torch.multiprocessing.set_sharing_strategy('file_system')


def nas(args):
    highlight_msg("Running {}".format(args.MODE))

    # init ctc-crf
    gpus = os.environ['CUDA_VISIBLE_DEVICES']
    gpus = list(range(len(gpus.split(','))))
    gpus = torch.IntTensor(gpus)
    dataLoc = args.dataLoc
    ctc_crf_base.init_env(
        "{}/data/den_meta/den_lm.fst".format(dataLoc), gpus)

    args.batchsize = args.batchsize if not args.debug else 32

    print("> Build model")
    model = build_model(args)
    print("model built. Size:{:.2f}M".format(count_parameters(model)/1e6))

    print("> Build NAS manager")
    manager = build_manager(model, args)
    print("manager and optimizer built.")

    if args.visualize:
        from toolkit.visualize import draw
        os.makedirs('figs', exist_ok=True)
        draw(manager.net.module.printGraph(), "figs/visual")
        print("> Visualized architecture saved to figs/visual.png")
        return

    if args.show:
        print("> Show graph")
        show_path(manager, args)
        return

    print("> Data prepare")
    if args.MFR:
        dir_dataset = "{}/data/nosub_".format(dataLoc)
        print("Read data from no subsampling dir.")
    else:
        dir_dataset = "{}/data/".format(dataLoc)
    
    if args.Pickle:
        data_format = "pickle"
        tr_originset = SpeechDataset(
            "{}{}/tr.{}".format(dir_dataset, data_format, data_format))
        test_originset = SpeechDataset(
            "{}{}/cv.{}".format(dir_dataset, data_format, data_format))
    else:
        data_format = "hdf5"
        tr_originset = SpeechDatasetMem(
            "{}{}/tr.{}".format(dir_dataset, data_format, data_format), args.debug)
        test_originset = SpeechDatasetMem(
            "{}{}/cv.{}".format(dir_dataset, data_format, data_format), args.debug)
    print("Data prepared.")

    # Rearrange the size of trainset and devset. The trainset-shared length
    # is relative to size of original trainset.
    if not args.debug and args.ratioTestShared > 0.:
        total_len = len(tr_originset)+len(test_originset)
        assert args.ratioTestShared >= len(
            test_originset)/total_len, "{}".format(len(test_originset)/total_len)

        len_dev_shared = int(args.ratioTestShared *
                             total_len) - len(test_originset)
        print("Split original dataset to [{},{}] for training and testing respectively.".format(
            total_len-len_dev_shared-len(test_originset), len_dev_shared+len(test_originset)))

        tr_set, test_set = torch.utils.data.random_split(
            tr_originset, [len(tr_originset)-len_dev_shared, len_dev_shared])
        test_set = torch.utils.data.ConcatDataset([test_set, test_originset])
    else:
        tr_set, test_set = tr_originset, test_originset

    trainloader = DataLoader(tr_set, batch_size=args.batchsize,
                             shuffle=True, num_workers=4, collate_fn=PadCollate())
    testloader = DataLoader(test_set, batch_size=args.batchsize,
                            shuffle=True, num_workers=4, collate_fn=PadCollate())

    if args.MODE == "warmup":
        manager.warmUp(trainloader, testloader, args)

    if args.MODE == "archsearch":
        manager.archSearch(trainloader, testloader, args)

    if args.MODE == "retrain":
        del trainloader
        del testloader
        trainloader = DataLoader(tr_originset, batch_size=args.batchsize,
                                 shuffle=True, num_workers=4, collate_fn=PadCollate())
        testloader = DataLoader(test_originset, batch_size=args.batchsize,
                                shuffle=False, num_workers=4, collate_fn=PadCollate())
        manager.retrain(trainloader, testloader, args)

    ctc_crf_base.release_env(gpus)


def show_path(manager, args):
    manager.displayArch(showall=True, toFile=args.disptofile)
    print(manager.net.module.printGraph())


def check_dirorfile(*items):
    for item in items:
        if item is not None:
            assert item != '', "Receive an empty path!"
            assert os.path.exists(item), "{} doesn't exist!".format(item)


def nandetect(x: torch.Tensor) -> bool:
    return torch.isnan(torch.sum(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="recognition argument")
    parser.add_argument("--lamb", type=float, default=0.1,
                        help="Weight factor of CTC in CTC-CRF loss function.")
    parser.add_argument("--batchsize", type=int, default=128,
                        help="Batch size for warmup/arch search/retrain.")
    parser.add_argument("--lrNetParams", type=float, default=0.001,
                        help="Learning rate of network parameters during warmup/arch search.")
    parser.add_argument("--lrArchParams", type=float, default=0.001,
                        help="Learning rate of arch parameters during arch search.")
    parser.add_argument("--lrRetrain", type=float,
                        default=0.001, help="Learning rate for retrain.")
    parser.add_argument("--lossFnWarmup", type=str, default="ctc", choices=['ctc', 'crf'],
                        help="Loss function for warmup. Available: ctc;crf")
    parser.add_argument("--lossFnAS", type=str, default="ctc", choices=['ctc', 'crf'],
                        help="Loss function for arch search. Available: ctc;crf")
    parser.add_argument("--lossFnRetrain", type=str, default="crf", choices=['ctc', 'crf'],
                        help="Loss function for retrain. Available: ctc;crf")
    parser.add_argument("--seed", type=int, default=0,
                        help="Manual seed.")
    parser.add_argument("--minLossImprove", type=float, default=0.5,
                        help="Minumum loss imrove for warmup. 0. for arch search.")
    parser.add_argument("--ratioTestShared", type=float, default=0.,
                        help="Ratio to fetch from whole training dataset to new dev dataset.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to location of checkpoint for warmup/arch search/retrain.")
    parser.add_argument("--MODE", type=str, default='warmup', choices=['warmup', 'archsearch', 'retrain'],
                        help="Configure the NAS to mode:warmup;archsearch;retrain")
    parser.add_argument("--MFR", action="store_true",
                        help="Configure to Mixed Frame Rate.")    
    parser.add_argument("--show", action="store_true",
                        help="Show the candidates operations w.r.t arch weight probabilities.")
    parser.add_argument("--debug", action="store_true",
                        help="Configure to debug settings, would overwrite most of the options.")
    parser.add_argument("--Pickle", action="store_true",
                        help="Read from pickle data, defaultly hdf5 instead.")
    parser.add_argument("--nasConfig", type=str, default=None,
                        help="Path to configuration file of NAS.")
    parser.add_argument("--dataLoc", type=str, default=None,
                        help="Location of training/testing data.")
    parser.add_argument("--warmupDir", type=str, default=None,
                        help="Directory to save the log and model files in warmup.")
    parser.add_argument("--archsearchDir", type=str, default=None,
                        help="Directory to save the log and model files in arch search.")
    parser.add_argument("--retrainDir", type=str, default=None,
                        help="Directory to save the log and model files in retrain.")
    parser.add_argument("--resumeRetrain", type=str, default=None,
                        help="Path to location of checkpoint for resuming retrained model. Should be with the option --resume.")
    parser.add_argument("--disptofile", type=str, default=None,
                        help="Display probs of arch to specified file.")
    parser.add_argument("--visualize", action="store_true",
                        help="Display the derived architecture.")
    args = parser.parse_args()

    check_dirorfile("{}/data/den_meta/den_lm.fst".format(args.dataLoc), args.resume, args.nasConfig, args.resumeRetrain,
                    args.dataLoc, args.warmupDir, args.archsearchDir, args.retrainDir)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    if args.debug:
        highlight_msg("Debug mode.")
    if not args.show and not args.debug and args.resume is None:
        if args.MODE == "warmup":
            assert len(os.listdir(args.warmupDir)
                       ) == 0, "\"{}\" not clean.".format(args.warmupDir)
        elif args.MODE == "archsearch":
            assert len(os.listdir(args.archsearchDir)
                       ) == 0, "\"{}\" dir not clean.".format(args.archsearchDir)
        elif args.MODE == "retrain":
            assert len(os.listdir(args.retrainDir)
                       ) == 0, "\"{}\" not clean.".format(args.retrainDir)
        else:
            raise NotImplementedError

    nas(args)
