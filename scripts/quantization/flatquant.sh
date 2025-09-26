#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
tp=${2}     # 4
device=${3} # 0

model_name=$(basename "$model")

if [[ "$model_name" == "DeepSeek-R1-Distill-Qwen-1.5B" ]] || [[ "$model_name" == "DeepSeek-R1-Distill-Qwen-7B" ]]; then
    seqlen=4096
else
    seqlen=2048
fi

bits=("4" "8")
for BITS in "${bits[@]}"; do
    ASCEND_VISIBLE_DEVICES=${device} \
    python -m methods.flatquant.main \
        --model ${model} \
        --w_bits ${BITS} --a_bits ${BITS} --a_asym \
        --k_bits ${BITS} --k_asym --k_groupsize 128 \
        --v_bits ${BITS} --v_asym --v_groupsize 128 \
        --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
        --lwc --lac --cali_trans --add_diag \
        --cali_dataset wikitext2 --seqlen ${seqlen} \
        --output_dir ./outputs/modelzoo/flatquant/logs --save_matrix \
        --deactive_amp --direct_inv --tp ${tp} \
        --exp_name w${BITS}a${BITS}kv${BITS}tp${tp} \
        --save_qmodel_path ./outputs/modelzoo/flatquant/${model_name}-flatquant-w${BITS}a${BITS}kv${BITS}-tp${tp}
done
