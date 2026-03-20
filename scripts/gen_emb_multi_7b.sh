#!/bin/bash

# ============================================================================
# LLaVA 7B Multi-GPU FULL Dataset Embedding Script (Original Style)
# ============================================================================

# 전체 데이터를 몇 개의 청크로 나눌지 (원하시는 만큼 늘리셔도 됩니다)
num_shards=100
per_user=20
# 7B 모델은 인스턴스 하나당 4개의 GPU가 무조건 필요합니다.
# 8-GPU 서버이므로 "0,1,2,3" 번 GPU 세트와 "4,5,6,7" 번 GPU 세트를 묶어 2개의 작업만 동시에 돌아갈 수 있습니다.
gpus=("0,1,2,3" "4,5,6,7")

exp_num=0
dry_run=false
output_dir="data/emb_test_7b_full"
mkdir -p $output_dir

# 인자를 넘기면 해당 청크만 실행합니다 (ex: bash script.sh 0)
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments sequentially"
fi

for (( which_shard=0; which_shard<num_shards; which_shard++ )); do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi
    
    # 워커 분배 (exp_num에 따라 0번 세트 혹은 1번 세트의 4-GPU를 할당)
    which_gpu=${gpus[$exp_num % ${#gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$which_gpu

    echo "Running chunk $which_shard on GPUs $which_gpu"
    command="python llava_embeddings/pick_a_pick_user_emb_7b.py \
        --device_map auto \
        --pretrained lmms-lab/llava-onevision-qwen2-7b-ov-chat \
        --num_shots 2 \
        --num_chunks $num_shards \
        --which_chunk $which_shard \
        --per_user $per_user \
        --save_every 1 \
        --output_dir $output_dir \
    "
    
    echo $command
    if [ $dry_run = false ]; then
        eval $command
    fi
    exp_num=$((exp_num+1))
done
