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

#### 2.	Q-Learning 
![](https://iknow-pic.cdn.bcebos.com/8b82b9014a90f603c4db973a2912b31bb051ed60?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>注意！虽然用了maxQ(s2)来估计下一个s2状态，但还没有在状态s2作出任何的行为，s2的决策部分要等到更新完了以后再重新另外执行这一过程

![](https://iknow-pic.cdn.bcebos.com/f11f3a292df5e0fed37e827c4c6034a85edf726c?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>
        ϵ - greedy是用在决策上的一种策略，如ϵ=0.9时，说明90%的情况按Q表的最优值来选择行为，10%的时间使用随机选择行为；
        <br>α是学习效率，来决定这一次误差有多少要被学习，α<1
        <br>γ是对未来奖励的衰减值
 
 #### 3. SARSA算法
![](https://iknow-pic.cdn.bcebos.com/8601a18b87d6277feae0954538381f30e824fcd9?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>SARSA算法在S2这一步估计的动作也是接下来要做的动作，所以现实值会进行改动，去掉maxQ,改为实实在在的该动作的Q值
<br>
![](https://iknow-pic.cdn.bcebos.com/e824b899a9014c0812c98fd81a7b02087af4f4d9?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>SARSA算法：说到做到，行为策略和目标策略相同
<br>Q-Learning：说到不一定做到，行为策略和目标策略不同

#### 4. SARSA（λ）
λ其实是一个衰变值，让你知道离奖励越远的步可能并不是让你最快拿到奖励的步。所以我们现在站在宝藏所处的位置，回头看看我们所走的寻宝之路，离宝藏越近的脚步我们看得越清楚，越远的脚步越渺小很难看清。所以我们索性认为离宝藏越近的脚步越重要，越需要好好更新。和之前提到的奖励衰减值γ一样，λ是脚步衰减值，都是一个在0和1之间的数.
<br>![](https://iknow-pic.cdn.bcebos.com/63d9f2d3572c11df2f23185e732762d0f603c2d9?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>![](https://iknow-pic.cdn.bcebos.com/10dfa9ec8a136327eb1f654e818fa0ec09fac7d9?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>当λ=0：Sarsa(0)就变成了SARSA的单步更新：每次只能更新最近的一步
<br>当λ=1：Sarsa(1)就变成了SARSA的回合更新：对所有步更新的力度一样
<br>当λ在（0，1），则取值越大，离宝藏越近的步更新力度越大。以不同力度更新所有与宝藏相关的步
<br>SARSA(λ)的伪代码：
<br>![](https://iknow-pic.cdn.bcebos.com/63d0f703918fa0ec8535c370369759ee3c6ddbd9?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>SARSA(λ)是向后看的过程，经历了哪些步就要标记一下，标记方法有两种：
<br>![](https://iknow-pic.cdn.bcebos.com/3c6d55fbb2fb4316f8eddd0730a4462308f7d3d9?x-bce-process=image/resize,m_lfit,w_600,h_800,limit_1)
<br>Method 1(accumulating trace): 遇到state就加一，没有遇到衰减，没有封顶值（可能会有）
<br>Method 2(replacing trace): 遇到state就加一，没有遇到衰减，有封顶值，到达封顶值在遇到不可以再往上加了，只能保持在峰值。


 
