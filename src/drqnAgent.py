"""
This part of code is about the agent in MARL and is coded based on the tutorial:
https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import tensorflow as tf
from src.drqn import DRQN
# import keras

np.random.seed(1)
tf.set_random_seed(1)


class DRQNAgent:
    def __init__(self, _id, config):

        # agent parameters
        self.id = _id
        self.n_channel = config.n_pu
        self.neighbors = config.neighbors
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size

        # q-network parameters
        self.lr = config.learning_rate
        self.gamma = config.reward_decay
        self.epsilon_max = config.e_greedy
        self.belief_decay = config.belief_decay
        self.replace_target_iter = config.replace_target_iter

        # build DRQNetwork
        self.net = DRQN(self.id, config)
        self.net.build()

        self.learn_step_counter = 0 # total learning step
        self.chan_beliefs = np.zeros(self.n_channel) # initialize the belief value

        # initialize log data
        self.failure = 0
        self.success = 0
        self.n_chan_chosen = np.zeros(self.n_channel)

        # initialize replay memory [b, a, r, b_]
        self.memory = np.zeros((self.memory_size, self.n_channel * 2 + 2))
        self.memory_counter = 0

        # open session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_history = []

    def store_transition(self, observation, reward, action):

        # update stats
        if reward < 0:
            self.failure += 1
        else:
            self.success += 1
        self.n_chan_chosen[action] += 1

        belief = self.chan_beliefs
        # update the channel belief
        for i in range(self.n_channel):
            if self.chan_beliefs[i] > 0.5:
                self.chan_beliefs[i] = max(0.5, self.belief_decay * self.chan_beliefs[i])
            else:
                self.chan_beliefs[i] = min(0.5, 1 - self.belief_decay * (1 - self.chan_beliefs[i]))
        self.chan_beliefs[action] = observation
        belief_ = self.chan_beliefs

        transition = np.hstack((belief, [action, reward], belief_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self):
        # to have batch dimension when feed into tf placeholder
        observation = self.chan_beliefs[np.newaxis, :]

        # choose action with epsilon-greedy policy
        if np.random.uniform() < self.epsilon_max:
            actions_value = self.sess.run(self.net.q_eval, feed_dict={self.net.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_channel)
        return action

    def learn(self):
        # check to replace the parameters of target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.net.replace_target_op)
            # print('successfully replace params in target net\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.net.q_next, self.net.q_eval],
            feed_dict={
                self.net.s_: batch_memory[:, -self.n_channel:],
                self.net.s: batch_memory[:, :self.n_channel],
            }
        )

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_channel].astype(int)
        reward = batch_memory[:, self.n_channel + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train evaluate_net
        _, self.cost = self.sess.run(
            [self.net._train_op, self.net.loss],
            feed_dict={
                self.net.s: batch_memory[:, :self.n_channel],
                self.net.q_target: q_target
            }
        )
        self.cost_history.append(self.cost)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def stat(self):
        print('su%d: successful sensing ratio %f' % (self.id, self.success / (self.success + self.failure)))
        print(self.n_chan_chosen/np.sum(self.n_chan_chosen))


