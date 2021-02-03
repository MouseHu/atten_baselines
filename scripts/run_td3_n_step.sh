#!/usr/bin/env bash
#declare -a gpus=(7 1 2)
#declare -a envs=("Ant-v2" "HalfCheetah-v2"  "Swimmer-v2")
#declare -a envs_alias=("ant" "halfcheetah" "swimmer")
declare -a gpus=(3 4 5)
declare -a envs=("Hopper-v2" "Walker2d-v2" "Humanoid-v2")
declare -a envs_alias=("hopper" "walker" "humanoid")
#export PYTHONPATH=/home/hh/attention_baselines/
export PYTHONPATH=/home/hh/atten_baselines/
for ((i = 0 ; i < ${#gpus[@]} ; i++)); do
    for ((seed = 0 ; seed < 5 ; seed++));do
        CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=/home/hh/ddq_ablation nohup python /home/hh/atten_baselines/run/run_td3.py --agent=TD3MemDDQ --beta=0.9 --alpha=0.5 --max_step=5 --comment=${envs_alias[$i]}_ddq_max_step=5_beta=0.9_$1_${seed} --env-id=${envs[$i]} > ./logs/${envs_alias[$i]}_ddq_max_step=5_beta=0.9_$1_${seed}.out &
    done
done