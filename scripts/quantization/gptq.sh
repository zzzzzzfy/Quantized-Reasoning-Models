#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
tp=${2}     # 4
device=${3} # 0

model_name=$(basename "$model")

bits=("3" "4")
for BITS in "${bits[@]}"; do
    CUDA_VISIBLE_DEVICES=${device} \
    python -m methods.quarot_gptq.save_fake_quant \
        --model ${model} \
        --w_bits ${BITS} --w_clip --w_asym --w_groupsize 128 --act_order \
        --tp ${tp} \
        --save_qmodel_path ./outputs/modelzoo/gptq/${model_name}-gptq-w${BITS}g128-tp${tp} \
        --cal_dataset reasoning-numina-math-1.5
done
