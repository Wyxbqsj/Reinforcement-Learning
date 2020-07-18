import numpy as np
import pandas as pd 

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
        if np.random.rand()<self.epsilon:
            #choose best action
            state_action=self.q_table.loc[observation,:]
            action=np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            #choose random action
            action=np.random.choice(self.actions)
        return action

    def learn(self,*args):
        pass


#backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,trace_decay=0.9):
        super(SarsaLambdaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)
        #除了继承父类的参数，SARSA(lambda)还有自己的参数
        #backward view,eligibility trace——sarsa(lambda)的新参数
        self.lambda_=trace_decay #脚步衰减值，在0-1之间
        self.eligibility_trace=self.q_table.copy() #和q_table一样的table，也是一个行为state，列为action的表，经历了某个state，采取某个action时，在表格对应位置加1

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            #生成一个符合q_table标准的全0数列
            to_be_append=pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            #追加在q_table后
            self.q_table=self.q_table.append(to_be_append)
            
            #追加在eligibility_trace后
            #also update eligibility trace
            self.eligibility_trace=self.eligibility_trace.append(to_be_append)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict=self.q_table.loc[s,a]
        if s_!='terminal':
            q_target=r+self.gamma*self.q_table.loc[s_,a_]
        else:
            q_target=r
        error=q_target-q_predict #求出误差，反向传递过去

        #increase trace amount for visited state_action pair
        #计算每个步的不可或缺性（eligibility trace）

        #Method 1:没有封顶值，遇到就加一
        self.eligibility_trace.loc[s,a]+=1

        #Method 2:有封顶值
        #self.eligibility_trace.loc[s,:]*=0 #对于这个state，把他的action全部设为0
        #self.eligibility_trace.loc[s,a]=1 #在这个state上采取的action，把它变为1

        #Q表update，sarsa(lambda)的更新方式：还要乘以eligibility_trace
        self.q_table+=self.lr*error*self.eligibility_trace

        #decay eligibility trace after update，体现eligibility_trace的衰减：lambda_是脚步衰变值，gamma是reward的衰变值
        self.eligibility_trace*=self.gamma*self.lambda_




