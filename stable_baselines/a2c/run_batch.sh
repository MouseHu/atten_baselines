#bash
batch_id=repr3

CUDA_VISIBLE_DEVICES=0 OPENAI_LOGDIR=~/attn_atari/${batch_id}1/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
  >log_${batch_id}1.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=~/attn_atari/${batch_id}2/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
 --repr_coef=0.1 >log_${batch_id}2.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=2 OPENAI_LOGDIR=~/attn_atari/${batch_id}3/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
 --lr=0.0001  > log_${batch_id}3.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=3 OPENAI_LOGDIR=~/attn_atari/${batch_id}4/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
 --repr_coef=0.1 --encoder_coef=0.001  > log_${batch_id}4.txt&
