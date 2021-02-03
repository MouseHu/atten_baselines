#!/usr/bin/env bash

#declare -a envs=("Ant-v2" "HalfCheetah-v2" "Hopper-v2" "Walker2d-v2" "Humanoid-v2" "Swimmer-v2")
#declare -a envs_alias=("ant" "halfcheetah" "hopper" "walker" "humanoid" "swimmer")
#declare -a gpus=(0 1 2)
#declare -a envs=("Ant-v2" "HalfCheetah-v2"  "Swimmer-v2")
#declare -a envs_alias=("ant" "halfcheetah" "swimmer")
declare -a gpus=(3 4 5)
declare -a envs=("Hopper-v2" "Walker2d-v2" "Humanoid-v2")
declare -a envs_alias=("hopper" "walker" "humanoid")

export PYTHONPATH=/home/hh/atten_baselines/
for ((i = 0 ; i < ${#gpus[@]} ; i++)); do
    for ((seed = 0 ; seed < 5 ; seed++));do
        CUDA_VISIBLE_DEVICES=${gpus[$i]} OPENAI_LOGDIR=/home/hh/ddq_ablations nohup python /home/hh/atten_baselines/run/run_td3.py --alpha=0.5 --beta=-1 --max_step=5 --evaluation --agent=TD3MemMany --comment=${envs_alias[$i]}_td3_amc_no_double_$1_$seed --env-id=${envs[$i]} > ./logs/${envs_alias[$i]}_td3_amc_no_double_$1_$seed.out &
    done
done