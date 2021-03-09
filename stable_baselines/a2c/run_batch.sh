#bash
batch_id=pong_contrasive_exp1
base_dir=/home/lzy/atten_baselines/experiments

CUDA_VISIBLE_DEVICES=6 OPENAI_LOGDIR=${base_dir}/${batch_id}1/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
 --repr_coef=0 --decoder_coef=0 --encoder_coef=0 --regularize_coef=0 >log_${batch_id}1.txt&

sleep 3s
CUDA_VISIBLE_DEVICES=6 OPENAI_LOGDIR=${base_dir}/${batch_id}2/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
 --repr_coef=0.05 --decoder_coef=0 --encoder_coef=0 --regularize_coef=0 >log_${batch_id}2.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=7 OPENAI_LOGDIR=${base_dir}/${batch_id}3/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
 --repr_coef=0.1 --decoder_coef=0 --encoder_coef=0 --regularize_coef=0 >log_${batch_id}3.txt&

sleep 3s

CUDA_VISIBLE_DEVICES=7 OPENAI_LOGDIR=${base_dir}/${batch_id}4/ nohup python run_atari_generalize.py --num-timesteps=20000000 \
 --repr_coef=0.3 --decoder_coef=0 --encoder_coef=0 --regularize_coef=0 >log_${batch_id}4.txt&