#bash
batch_id=norepr_tunepara

CUDA_VISIBLE_DEVICES=2 OPENAI_LOGDIR=~/atten_baselines/experiments/${batch_id}1/ nohup python run_atari_generalize.py --num-timesteps=40000000 \
 --lr=0.0003 >log_${batch_id}1.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=3 OPENAI_LOGDIR=~/atten_baselines/experiments/${batch_id}2/ nohup python run_atari_generalize.py --num-timesteps=40000000 \
 --vf_coef=1 >log_${batch_id}2.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=4 OPENAI_LOGDIR=~/atten_baselines/experiments/${batch_id}3/ nohup python run_atari_generalize.py --num-timesteps=40000000 \
 --vf_coef=0.25  > log_${batch_id}3.txt&


