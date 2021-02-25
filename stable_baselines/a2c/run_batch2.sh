#bash
batch_id=mlp_attn_contra_exp2
base_dir=/home/lzy/atten_baselines/experiments

CUDA_VISIBLE_DEVICES=2 OPENAI_LOGDIR=${base_dir}/${batch_id}1/ nohup python run_atari_generalize.py --num-timesteps=40000000 \
 --repr_coef=0.3 >log_${batch_id}1.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=3 OPENAI_LOGDIR=${base_dir}/${batch_id}2/ nohup python run_atari_generalize.py --num-timesteps=40000000 \
 --repr_coef=1 >log_${batch_id}2.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=4 OPENAI_LOGDIR=${base_dir}/${batch_id}3/ nohup python run_atari_generalize.py --num-timesteps=40000000 \
 --repr_coef=2 >log_${batch_id}3.txt&