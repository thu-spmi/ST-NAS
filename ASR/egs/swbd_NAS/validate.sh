#!/bin/bash

. ./cmd.sh
. ./path.sh
dir=`pwd -P`
model="models/retrain/model.bestforinfer.pt"

data_eval2000=data/eval2000
ark_dir=exp/decode_eval2000/ark

feats_eval2000="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_eval2000/utt2spk \
  scp:$data_eval2000/cmvn.scp scp:$data_eval2000/feats.scp ark:- | add-deltas ark:- ark:- |" 

mkdir data/test_data
copy-feats "$feats_eval2000"   ark,scp:data/test_data/eval2000.ark,data/test_data/eval2000.scp

for set in eval2000; do
  mkdir -p exp/decode_$set/ark
  python3 scripts/calculate_logits.py --resume=$model --nj=20 --input_scp=data/test_data/${set}.scp \
  --output_unit=46 --data_path=$dir --output_dir=$ark_dir
done

graphdir=data/lang_phn_sw1_tg
lat_dir=exp/decode_eval2000/lattice_sw1_tg
scoring_opts=

mkdir -p $lat_dir

$cmd JOB=1:20 $ark_dir/log/decode.JOB.log \
  latgen-faster --max-mem=200000000 --min-active=200 --max-active=7000 --beam=17.0 --lattice-beam=8.0 \
  --minimize=false --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $graphdir/TLG.fst "ark:$ark_dir/decode.JOB.ark" "ark:|gzip -c > $lat_dir/lat.JOB.gz" || exit 1
local/score_sclite.sh $scoring_opts --cmd "$cmd" $data_eval2000 $graphdir $lat_dir
echo "score confidence and timing with sclite"

nj=20
local/lmrescore_const_arpa.sh --cmd "$cmd" data/lang_phn_sw1_{tg,fsh_fg} data/eval2000 exp/decode_eval2000/lattice_sw1_{tg,fsh_fg} $nj || exit 1;

grep Sum exp/decode_eval2000/lattice_sw1_fsh_fg/score_*/eval2000.ctm.filt.sys | utils/best_wer.sh
grep Sum exp/decode_eval2000/lattice_sw1_fsh_fg/score_*/eval2000.ctm.callhm.filt.sys | utils/best_wer.sh
grep Sum exp/decode_eval2000/lattice_sw1_fsh_fg/score_*/eval2000.ctm.swbd.filt.sys | utils/best_wer.sh
