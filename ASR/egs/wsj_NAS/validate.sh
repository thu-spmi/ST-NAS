#!/bin/bash

. ./cmd.sh
. ./path.sh
dir=`pwd -P`
model='models/retrain/model.bestforinfer.pt'

for set in dev93 eval92; do
    mkdir -p exp/decode_$set/ark
    ark_dir=exp/decode_$set/ark
    CUDA_VISIBLE_DEVICES="0" python3 scripts/calculate_logits.py --resume=$model --nj=20 --input_scp=data/test_data/${set}.scp --output_unit=72 --data_path=$dir --output_dir=$ark_dir || exit 1
done

acwt=1.0
nj=20
for set in dev93 eval92; do
   for lmtype in tgpr bd_tgpr; do
       bash local/decode.sh $acwt $set  $nj ${lmtype}
   done
done

for set in dev93 eval92; do
  mkdir -p exp/decode_${set}/lattice_bd_fgconst
  ./local/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_phn_test_bd_{tgpr,fgconst} data/test_${set} exp/decode_${set}/lattice_bd_{tgpr,fgconst} $nj || exit 1;
  mkdir -p exp/decode_${set}/lattice_tg
  ./local/lmrescore.sh --cmd "$decode_cmd" data/lang_phn_test_{tgpr,tg} data/test_${set} exp/decode_${set}/lattice_{tgpr,tg} $nj || exit 1;
done

grep WER exp/decode_eval92/lattice_bd_fgconst/wer_* | utils/best_wer.sh
grep WER exp/decode_dev93/lattice_bd_fgconst/wer_* | utils/best_wer.sh
