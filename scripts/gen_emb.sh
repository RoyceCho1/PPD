num_shards=8
per_user=20
which_shards=(0 1 2 3 4 5 6 7)
gpus=(0 1 2 3 4 5 6 7)

exp_num=0
dry_run=false
output_dir="data/"
mkdir -p $output_dir

which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments"
fi

for which_shard in ${which_shards[@]}; do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi
    which_gpu=${gpus[$exp_num % ${#gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$which_gpu

    echo "Running experiment $exp_num on GPU $which_gpu"
    command="python llava_embeddings/pick_a_pick_user_emb.py \
        --num_chunks $num_shards \
        --which_chunk $which_shard \
        --per_user $per_user \
        --output_dir $output_dir \
    "
    
    echo $command
    if [ $dry_run = false ]; then
        eval $command
    fi
    exp_num=$((exp_num+1))
done