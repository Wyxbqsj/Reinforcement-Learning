from maze_env import Maze
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        observation=env.reset() #初始化观测值

        while True:
            env.render() #渲染刷新环境
            action=RL.choose_action(str(observation))

            observation_,reward,done=env.step(action)

            RL.learn(str(observation),action,reward,str(observation_))

            observation=observation_

            if done:
                break
    #end of the game
    print('game over')
    env.destroy()

if __name__=="__main__":
    env=Maze()
    RL=QLearningTable(actions=list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()

        

