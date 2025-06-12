eval "$(conda shell.bash hook)"
conda activate rlhf

batch_size=64
lrs=(1e-3 1e-4 1e-5 1e-6)
num_layers=(4)
dropouts=(0.0 0.1)
gpus=(0 1 2 3 4 5 6 7)

exp_num=0
dry_run=false
debug=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments"
fi
if [ $debug = true ]; then
    export WANDB_MODE=dryrun
fi

for lr in ${lrs[@]}; do
for num_layer in ${num_layers[@]}; do
for dropout in ${dropouts[@]}; do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi
    which_gpu=${gpus[$exp_num % ${#gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$which_gpu

    echo "Running experiment $exp_num on GPU $which_gpu"
    command="python user_classification/user_classifier.py \
        --learning_rate $lr \
        --num_layers $num_layer \
        --dropout $dropout \
        --batch_size $batch_size \
    "
    
    echo $command
    if [ $dry_run = false ]; then
        eval $command
    fi
    exp_num=$((exp_num+1))
done
done
done