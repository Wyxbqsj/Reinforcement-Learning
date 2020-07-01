import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions=actions 
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=e_greedy
        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64)

    def choose_action(self,observation):
        self.check_state_exist(observation) #判断当前观测值是否在表中

        #动作选择
        if np.random.uniform()<self.epsilon:#numpy.random.uniform(x,y)随机生成一个浮点数，它在 [x, y] 范围内，默认值x=0,y=1
            #choose best action
            state_action=self.q_table.loc[observation,:]
            #some actions may have the same value,randomly choose in these actions
            action=np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            #choose random action
            action=np.random.choice(self.actions)
        return action
    
    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        q_predict=self.q_table.loc[s,a]
        if s_!='terminal': #next state is not terminal
            q_target=r+self.gamma*self.q_table.loc[s_,:].max() 
        else: #到达terminal，得到奖励
            q_target=r
        self.q_table.loc[s,a]+=self.lr*(q_target-q_predict) #更新

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            #不在，就将新出现的state值追加到表中
            self.q_table=self.q_table.append(
                pd.Series( #Series是能够保存任何类型的数据(整数，字符串，浮点数，Python对象等)的一维标记数组。轴标签统称为索引。
                    [0]*len(self.actions), #len()方法返回列表元素个数,[0]*3=[0,0,0]
                    index=self.q_table.columns,
                    name=state,
                )
        )


         
