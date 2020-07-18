"""
from maze_env1 import Maze 
from RL_brain1 import SarsaTable

def update():
    for episode in range(100):
        observation=env.reset() #从环境里获得observation
        action=RL.choose_action(str(observation)) 
        #Q-Learning的action是在下面这个while循环里选的，SARSA算法是在循环外
        while(True):
            env.render() #环境更新
            observation_,reward,done=env.step(action)
            action_=RL.choose_action(str(observation_))
            #与Q—learning不同之处：SARSA还要传入下一个动作action_,而Q—learning不需要
            RL.learn(str(observation),action,reward,str(observation_),action_)
            
            #sarsa所估计的下一个action，也是sarsa会采取的action
            #observation和action都更新
            observation=observation_
            action=action_

            if done:
                break
    #end of the game
    print('game over')
    env.destroy()

if __name__=="main":
    env=Maze()
    RL=SarsaTable(actions=list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()
"""
from maze_env1 import Maze
from RL_brain1 import SarsaTable


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation and action
            observation = observation_
            action = action_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()