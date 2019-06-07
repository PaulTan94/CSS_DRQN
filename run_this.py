"""
In this version, ...
"""

from radio_env import Radio
from MARL import Agent
import time


def run_radio():
    # iteration time
    iteration = 10000
    for timestep in range(iteration):
        # fresh env
        env.render()

        # Each time slot
        for k in range(len(su)):
            # choose action based on the channel beliefs
            action = su[k].choose_action()

            # SUs take action and get observation and reward
            observation, reward = env.observe(action, k)
            su[k].store_transition(observation, reward, action)

            # learn
            if timestep % 5 == 0:
                su[k].learn()

            # if timestep > 9500:
            #     time.sleep(0.2)

        env.step(timestep)
        timestep += 1

    print("Simulation over")
    env.destroy()


if __name__ == "__main__":
    # create radio environment and sensing agents
    env = Radio()
    n_su = 9
    su = []
    for i in range(n_su):
        su.append(Agent(i, env.n_pu, env.neighbors[i]))

    env.after(100, run_radio)
    env.mainloop()

    for i in range(n_su):
        su[i].stat()

    su[0].plot_cost()
    env.stat()
