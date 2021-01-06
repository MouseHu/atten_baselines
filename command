CUDA_VISIBLE_DEVICES=1 OPENAI_LOGDIR=/home/lzy/experiments/coin5/ python run_gridworld_generalize.py --id=coin5 --num-timesteps=1500000
nohup tensorboard --logdir=./ --port=6007 &
git push origin main:lzy_test
