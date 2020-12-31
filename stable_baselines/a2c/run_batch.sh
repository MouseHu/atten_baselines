#bash
batch_id=breakout_repr

CUDA_VISIBLE_DEVICES=4 OPENAI_LOGDIR=~/attn_atari/${batch_id}1/ nohup python run_atari_generalize.py --num-timesteps=10000000 \
 --repr_coef=1 >log_${batch_id}1.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=5 OPENAI_LOGDIR=~/attn_atari/${batch_id}2/ nohup python run_atari_generalize.py --num-timesteps=10000000 \
 --repr_coef=1 > log_${batch_id}2.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=6 OPENAI_LOGDIR=~/attn_atari/${batch_id}3/ nohup python run_atari_generalize.py --num-timesteps=10000000\
 --repr_coef=1 >log_${batch_id}3.txt&