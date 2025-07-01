#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
tp=${2}     # 4
device=${3} # 0,1,2,3

model_name=$(basename "$model")

BITS=8

CUDA_VISIBLE_DEVICES=${device} \
python -m methods.smoothquant.save_fake_quant \
    --model ${model} \
    --w_bits ${BITS} --w_clip \
    --a_bits ${BITS} --a_asym \
    --k_bits ${BITS} --k_asym --k_groupsize 128 \
    --v_bits ${BITS} --v_asym --v_groupsize 128 \
    --seqlen 2048 --nsamples 128 --cal_dataset reasoning-numina-math-1.5 \
    --tp ${tp} \
    --save_qmodel_path ./outputs/modelzoo/smoothquant/${model_name}-smoothquant-w${BITS}a${BITS}kv${BITS}-tp${tp}
