# Reinforcement-Learning
《白话强化学习》笔记+莫烦Python RL教程代码
<br>[莫烦Code](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
<br>[莫烦Python](https://morvanzhou.github.io)
#### 1.	强化学习方法分类汇总：
（1）`不理解环境（Model-Free RL）`：不尝试理解环境，环境给了什么就是什么；机器人只能按部就班一步一步等待真实世界的反馈，再根据反馈采取下一步的行动  <br>理解环境（Model-Based RL）：学会了用一种模型来模拟环境；能够通过想象来预判断接下来要发生的所有情况，然后根据这些想象中的情况选择最好的那种，并根据这种情况来采取下一步的策略。
<br>（2）`基于概率（Policy-Based RL）`：通过感官分析所处的环境，直接输出下一步采取的各种行动的概率，然后根据概率采取行动，所以每种动作都有可能被选中，只是可能性不同；用一个概率分布在连续动作中选择特定的动作
<br>`基于价值（Value-Based RL）`：通过感官分析所处的环境，直接输出所有动作的价值，我们会选择价值最高的那个动作；对于连续的动作无能为力 
<br>（3）`回合更新（Monte-Carlo update）`：假设强化学习是一个玩游戏的过程。游戏开始后需要等待游戏结束，然后再总结，再更新我们的行为准则
<br>`单步更新（Temporal-Difference update）`：在游戏进行中的每一步都在更新，不用等待游戏的结束，这样就能边玩边学习了
<br>（4）`在线学习（On-Policy）`：本人在场，而且必须是本人边玩边学习
<br>`离线学习（Off-Policy）`：可以选择自己玩，也可以选择看着别人玩，通过看着别人玩来学习别人的行为准则，同样是从过往经历中学习，但这些经历没必要是自己的

#### 2.	Q-Learning (Off-line)
![](https://iknow-pic.cdn.bcebos.com/8b82b9014a90f603c4db973a2912b31bb051ed60?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>注意！虽然用了maxQ(s2)来估计下一个s2状态，但还没有在状态s2作出任何的行为，s2的决策部分要等到更新完了以后再重新另外执行这一过程

![](https://iknow-pic.cdn.bcebos.com/f11f3a292df5e0fed37e827c4c6034a85edf726c?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>
        ϵ - greedy是用在决策上的一种策略，如ϵ=0.9时，说明90%的情况按Q表的最优值来选择行为，10%的时间使用随机选择行为；
        α是学习效率，来决定这一次误差有多少要被学习，α<1
        γ是对未来奖励的衰减值
