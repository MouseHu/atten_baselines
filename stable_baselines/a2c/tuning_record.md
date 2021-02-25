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
收敛速度快了一点点。

最后再试试标准网络，再不行就算了。发现模仿SB的标准extractor之后可以达到类似的performance（不管加不加attention结构），问题可能在初始化方式上？（甚至lr=5e-4,lr=7e-4跑出来结果都一样）

经过测试，真的是因为卷积层的初始化方式，太神奇了。

#### 第二阶段：加入repr
现在开始关注generalize的performance。

attn_exp
test1：默认参数再跑一遍。
generalize效果出人意料地不错

test2:repr=1
似乎完全放弃了对repr

test3:repr=0.1
训练慢了，generalize效果变好了，但attention似乎有时候不准

目前问题：
1.attention resize时位置不准，所以最好不要让masked_image全黑。
只需要调整normalize的分布。
需要一个map把[0,1]映射到[0.3,1]，y=0.2+0.8x

2.同一时间的不同东西分布在不同目录，看起来很不方便。所以重新按时间排序。

3.对游戏了解不够。自己要玩一下。
发现了有的图像会有不同颜色物体的拉长，可能是因为对游戏图像预处理的结果，不深究。

新的测试：在之前test3的基础上稍微调调参，重复一下.

2.14
重新测了一下，发现seed的影响还挺大的，且加了attention后trainning-performance可能会变差，test-performance似乎会变好但不是很稳定。
现在的attention仍然不是很准，范围倒还合适，所以loss系数基本不用调。不过contrasive系数好像太大了，调小点

按理来说，attention应该是寻找内部不同位置feature之间的关系，但是现在的attention生成方式只依靠于local的feature，我暂时认为不需要长程依赖，但是改成mlp是必要的。
如果再不行可以试试这个:https://blog.csdn.net/qq_39478403/article/details/105455877
需要把attention subnetwork搞复杂一点，且多跑几个seed。

现在看来各种loss平衡性都还可以，所以就单纯调一下网络结构试试吧~

#### 2.15
MLP attention的话，确实generalize变好了，performance变差了。
感觉是contrasive loss在起作用，但是attenttion仍然不准，再用新的attention结构试试。

#### 2.16
汇报内容：
1.确实是因为初始化方式，现在可以达到同样的训练水平

2.不加repr generalize也可能很好
很奇怪，应该是环境不够对抗。

3.attention不准
解决方案：试试多加几层1X1卷积，或者用那篇论文
https://blog.csdn.net/a312863063/article/details/83552111

反馈
1.wrapper：find difficult envs

2.attention：
- padding方式（排除）
- ？
边缘+亮点有问题
在特别简单的时候不容易attend到边缘。
regularzation sparse，卷积在边缘有重叠
换一个结构 or simplify

3. 分开使用attention_loss and contrasive and decoder
先做对比实验，弄清楚哪个loss work。
问题在于怎么加入regularzation，training performance能上去。
performance下降和generalization gap的减小是为什么？

2.18
self-attention那篇论文中的方法直接失效了，暂时先用mlp-attention，换个环境试试
另外，由于中毒，数据盘无了，需要重新跑一下baseline

从constant-rectangle换成diagonals试试。

2.19
diagnals好难（就左边长的斜对角线），所有seed的reward都收敛到2-6了，换一个。
and repr_coef>=1 会导致contrasive_loss不收敛，以后大致没有尝试的必要了。
0.1 or 0.3时收敛比较快。
今天快速试一下其他环境.

2.22
环境：
rec过于简单.
gl也许刚好，但只用contrasive loss很难学出来.
有一个加contrasive loss的seed效果还不错，不知道能不能复现。
下一步：
0.先试试加入value-prediction效果会不会变好
1.选择attention subnetwork,尝试mlp(现在没必要用attention,因为performance还不够好)
2.改变环境wrapper，利用颜色区分peddle and green line.
3.在gl环境中加入各种loss

应该先调gen的performance然后再加attention.

有几个只用contrasive loss还不错。
根据已有结果
repr_coef定为0.05。
contra_coef定为0.01。

2.25
进展：
顺序:先加入contrasive loss，然后是decoder loss，最后encoder loss
1.找到了比较好的环境（解决了zero-shot结果过于好的问题）
diagnol过难，rec过于简单，gl刚刚好

2.contrasive loss学到了一定的内容

问题：环境复杂度够吗？
可以用四个共享参数的network分别attend四帧。
现有方法不work环境复杂度就差不多。

问题2：attention到底是为了可视化还是提高performance？
regularzation ，对噪声更鲁棒。
给feature map的相关层加regularzation。

问题3：在a2c策略下，value prediction的意义是？
TD可能会带来训练不稳定。


预计打算：
1.改网络结构。
2.加regularzation。
3.继续细调decoder和encoder loss，直到attention正确为止。

建议：
1.多读paper
2.多交流
3.养成文献管理的习惯
4.实验还好，最重要的是有条理，控制好变量。







