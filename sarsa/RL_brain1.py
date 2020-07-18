"""
import numpy as np
import pandas as pd

#Q-Learning和SARSA的公共部分写在RL class内，让他们俩继承
class RL(object):
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions=action_space #a list
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=e_greedy

        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64)

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table=self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

        
    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.rand()<self.epsilon:  #np.random.rand()可以返回一个服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)
            #choose best action
            state_action=self.q_table.loc[observation,:]
            #some action may have the same value, randomly choose on in these actions
            action=np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            #choose random action
            action=np.random.choice(self.actions)
        return action

    def learn(self,*args): #Q-Learning和SARSA的这个部分不一样，接受的参数也不一样
        pass

#off-policy
class QLearningTable(RL): #继承了class RL
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(QLearningTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)
    
    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        q_prediect=self.q_table.loc[s,a]
        if s_!='terminal': #next state isn't terminal
            q_target=r+self.gamma*self.q_table.loc[s_,:].max() #找出s_下最大的那个动作值
        else: #next state is terminal
            q_target=r
        self.q_table.loc[s,a]+=self.lr*(q_target-q_prediect) #update

#on-policy 边学边走，比Q-Learning要胆小一点的算法
class SarsaTable(RL): ##继承了class RL
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(SarsaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)
    
    def learn(self,s,a,r,s_,a_): #比Q-learning多一个a_参数
        self.check_state_exist(s_)
        q_prediect=self.q_table.loc[s,a]
        if s_!='terminal':
            q_target=r+self.gamma*self.q_table.loc[s_,a_] #具体的s_,a_确定的唯一动作值
        else:
            q_target=r;
        self.q_table.loc[s,a]+=self.lr*(q_target-q_prediect)
"""
import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
            
