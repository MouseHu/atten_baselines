## 调参记录
*李正远*

- 这是一份对run_atari_generalize.py的调参记录，在调完之后可以被删除

- 待调参数：lr,repr_coef,atten_encoder_coef,atten_decoder_coef

#### 相关参数计算方式

self.repr_loss = self.contrastive_loss + self.encoder_loss + self.decoder_loss + self.recon_coef * self.reconstruct_loss
a2c_loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

#### 我也不知道是什么
repr_coef=1,begin_repr=0,learning_rate=7e-4,atten_encoder_coef=5/256
atten_decoder_coef=1.
regularize_coef=1e-4
在此情况，encoder_loss为0.02-0.1之间，contrasive_loss可以训到1e-3，decoder_loss约为1e-3,reconstruct_loss约为0.02,value_loss在0.05-0.03之间，policy_loss在0.06--0.06之间
reward的话，加repr平均为1.8,不加平均为1.2
attention总是会变成全黑的，似乎应该把系数调低一些,对啊，encoder_loss那么大，显然不合适！调低十倍试试

奇怪了，重新测试一下吧，把lr调到1e-4，其他不动，加长训练步数看看.
总觉得代码好像有问题，决定查一下代码，因为PPO2收敛非常快

#### 第一阶段：norepr
先拿不加repr的版本调一调,结果还不错,不过明显慢于PPO2
参数：lr=2.5e-4,vf_coef=0.5,其余不变。以此作为默认参数。

结果：10M step，reward=30；4M step reward=5

--repr_coef=1
--repr_coef=0.1
--repr_coef=0.01
都不太行,不如先看看原来的结果能不能复现。

原始版本：600 step 到51
lr=1e-4:也能收敛，但是慢
--repr_coef=0.1:更慢了
--repr_coef=0.1 --encoder_coef=0.001：encoder提高了3倍，结果是几乎不work

不加repr调lr:
lr=2e-4，也能收敛，但慢
lr=5e-4、7e-4不能收敛

调lr和vf:
vf_coef=0.5 lr=2.5e-4 20M step reward_mean=84.3
vf_coef=1 直接不收敛了
vf_coef=0.25 20M step reward_mean=123
lr=3e-4 20M step reward_mean=224

baseline 20M step reward_mean=465,而且早就收敛了

lr=3e-4 vf_coef=0.25,20M step reward_mean 244

不work的：
--lr=0.0003 --vf_coef=0.2 
--lr=0.0003 --vf_coef=0.3 
work的：
--lr=0.00035 --vf_coef=0.25 20M step：312
--lr=0.00027 --vf_coef=0.25 20M step：321

所以最后的参数是:
lr=3.5e-4
vf_coef=0.25

收敛还是比标准网络慢很多，我怀疑是extractor里面最后一层变成了16的原因，调成64试试
#### 第二阶段：加入repr




