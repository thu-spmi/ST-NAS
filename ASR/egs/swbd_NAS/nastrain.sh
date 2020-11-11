#!/bin/bash

dir="."
stage=1

mkdir -p models/warmup
mkdir -p models/as
mkdir -p models/retrain

# warmup
if [ $stage -le 1 ]; then
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python3 scripts/ArchSearch.py \
--MODE="warmup" \
--batchsize=128 \
--lrNetParams=0.001 \
--lrArchParams=0.005 \
--ratioTestShared=0.05 \
--lossFnWarmup="ctc" \
--seed=0 \
--MFR \
--minLossImprove=0.5 \
--nasConfig="./config.json" \
--dataLoc=$dir \
--warmupDir="models/warmup/" \
--Pickle \
|| exit 1
fi

# arch search
if [ $stage -le 2 ]; then
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python3 scripts/ArchSearch.py \
--MODE="archsearch" \
--batchsize=64 \
--lrNetParams=0.001 \
--lrArchParams=0.005 \
--ratioTestShared=0.05 \
--lossFnAS="ctc" \
--seed=0 \
--MFR \
--minLossImprove=0.5 \
--nasConfig="./config.json" \
--dataLoc=$dir \
--archsearchDir="models/as/" \
--resume="models/warmup/model.bestforas.pt" \
--Pickle \
|| exit 1
fi

# retrain
if [ $stage -le 3 ]; then
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python3 scripts/ArchSearch.py \
--MODE="retrain" \
--lamb=0.1 \
--batchsize=128 \
--lrRetrain=0.001 \
--lossFnRetrain="crf" \
--seed=0 \
--MFR \
--nasConfig="./config.json" \
--dataLoc=$dir \
--retrainDir="models/retrain/" \
--resume="models/as/model.bestforretrain.pt" \
--Pickle \
|| exit 1
fi
