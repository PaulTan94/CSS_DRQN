"""
In this version, ...
"""

from src.radio_env import Radio
from src.drqnAgent import DRQNAgent
from src.config import ConfigAgent, ConfigEnv


def run_radio():
    # iteration time
    iteration = 10000
    for timestep in range(iteration):
        # fresh env
        env.render()

        # Each time slot
        for k in range(config_agent.n_su):

            action = su[k].choose_action()                       # agent chooses action based on the channel beliefs
            observation, reward = env.observe(action, k)         # agent takes action, gets observation and reward
            su[k].store_transition(observation, reward, action)  # agent stores transition

            # learn
            if timestep % 5 == 0:
                su[k].learn()

        # update radio environment
        env.step(timestep)
        timestep += 1

    print("Simulation over")
    env.destroy()


if __name__ == "__main__":

    # create radio environment
    config_env = ConfigEnv()
    env = Radio(config_env)

    # create agents
    config_agent = ConfigAgent()
    su = []
    for i in range(config_env.n_su):
        su.append(DRQNAgent(i, config_agent))

    env.after(100, run_radio)
    env.mainloop()

    for i in range(config_env.n_su):
        su[i].stat()

    su[0].plot_cost()
    env.stat()
