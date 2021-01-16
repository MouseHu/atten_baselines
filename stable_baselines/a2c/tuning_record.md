这是一份对run_atari_generalize.py的调参记录，在调完之后可以被删除

待调参数：lr,repr_coef,atten_encoder_coef,atten_decoder_coef

self.repr_loss = self.contrastive_loss + self.encoder_loss + self.decoder_loss + self.recon_coef * self.reconstruct_loss
a2c_loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

repr_coef=1,begin_repr=0,learning_rate=7e-4,atten_encoder_coef=5/256
atten_decoder_coef=1.
regularize_coef=1e-4
在此情况，encoder_loss为0.02-0.1之间，contrasive_loss可以训到1e-3，decoder_loss约为1e-3,reconstruct_loss约为0.02,value_loss在0.05-0.03之间，policy_loss在0.06--0.06之间
reward的话，加repr平均为1.8,不加平均为1.2
attention总是会变成全黑的，似乎应该把系数调低一些,对啊，encoder_loss那么大，显然不合适！调低十倍试试

奇怪了，重新测试一下吧，把lr调到1e-4，其他不动，加长训练步数看看.
总觉得代码好像有问题，决定查一下代码，因为PPO2收敛非常快

先拿不加repr的版本调一调,结果还不错,不过明显慢于PPO2
参数：lr=2.5e-4,vf_coef=0.5,其余不变。
结果：10M step，reward=30；4M step reward=5

不妨再调一下这两个参数,看能不能收敛快点，然后再加attention,另外注意，step稍微再多一点，每个point应该至少有2个seed
vf=0.5
lr=1e-4:12
lr=5e-4:34

lr=2.5e-4
vf=0.25
vf=1

发现代码写错了，重来一次。算了，跑太慢了，没必要，直接开始调带repr的版本吧

--repr_coef=1
--repr_coef=0.1
--repr_coef=0.01
都不太行,不如先看看原来的结果能不能复现。

原始版本：600 step 到51
lr=1e-4:也能收敛，但是慢
--repr_coef=0.1:更慢了
--repr_coef=0.1 --encoder_coef=0.001：encoder提高了3倍，结果是几乎不work




