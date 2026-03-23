#!/bin/bash

# ============================================================================
# LLaVA 7B Multi-GPU FULL Dataset Embedding Script (2-way Parallel / 2-GPU each)
# ============================================================================

# 전체 데이터를 몇 개의 청크로 나눌지
num_shards=100
per_user=20

# Option B: 2개의 Worker를 2장씩 묶어 동시 실행 (GPU 0,1번 1세트 / GPU 2,3번 1세트)
gpus=("0,1" "2,3")

dry_run=false
output_dir="data/emb_test_7b_full"
mkdir -p $output_dir

echo "Running all experiments 2-way parallel on 2-GPU groups: ${gpus[*]}"

# 진행 상황을 보기 쉽게 하기 위해
echo "--------------------------------------------------------"
echo "실시간으로 로그를 보시려면 다음 명령어를 병렬 창에서 사용하세요:"
echo "tail -f $output_dir/log_chunk_*.txt"
echo "--------------------------------------------------------"

for (( which_shard=0; which_shard<num_shards; which_shard++ )); do
    # 현재 청크 처리할 GPU 번호 계산 (0번 인덱스, 1번 인덱스 반복)
    gpu_idx=$((which_shard % 2))
    which_gpu=${gpus[$gpu_idx]}
    
    export CUDA_VISIBLE_DEVICES=$which_gpu
    
    echo "Running chunk $which_shard on GPU $which_gpu (Background)"
    
    # 2장씩 묶어 쓰므로 --load_4bit을 끄고 --device_map auto를 사용하여 메모리를 절반씩 분산시킵니다.
    command="python llava_embeddings/pick_a_pick_user_emb_7b.py \
        --device_map auto \
        --pretrained lmms-lab/llava-onevision-qwen2-7b-ov-chat \
        --num_shots 4 \
        --num_chunks $num_shards \
        --which_chunk $which_shard \
        --per_user $per_user \
        --save_every 1 \
        --output_dir $output_dir > $output_dir/log_chunk_${which_shard}.txt 2>&1"
    
    if [ "$dry_run" = false ]; then
        eval "$command &"
    else
        echo "$command &"
    fi
    
    # 2개의 백그라운드 프로세스가 런칭되었으면 대기
    if [ $(( (which_shard + 1) % 2 )) -eq 0 ]; then
        echo "Launched 2 parallel workers. Waiting for them to finish before starting the next batch..."
        wait
    fi
done

wait
echo "All chunks processed successfully! :)"
