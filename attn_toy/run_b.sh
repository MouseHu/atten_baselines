batch_id=norepr
CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/lzy/experiments/${batch_id}1/ nohup python run_gridworld_generalize.py --id=${batch_id}1 --num-timesteps=1500000 --seed=10 >log_${batch_id}1.txt&

CUDA_VISIBLE_DEVICES=2 OPENAI_LOGDIR=/home/lzy/experiments/${batch_id}2/ nohup python run_gridworld_generalize.py --id=${batch_id}2 --num-timesteps=1500000 --seed=20 >log_${batch_id}2.txt&

CUDA_VISIBLE_DEVICES=3 OPENAI_LOGDIR=/home/lzy/experiments/${batch_id}3/ nohup python run_gridworld_generalize.py --id=${batch_id}3 --num-timesteps=1500000 --seed=30 >log_${batch_id}3.txt&
