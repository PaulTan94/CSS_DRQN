"""
This part of code is about the agent in MARL and is coded based on the tutorial:
https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class Agent:
    def __init__(
            self,
            id_,
            n_channel,
            neighbors,
            learning_rate=0.01,
            reward_decay=0.9,
            belief_decay=0.9,
            e_greedy=0.95,
            replace_target_iter=100,
            memory_size=20,
            batch_size=20,
            # e_greedy_increment = None,
            # output_graph = Flase,
    ):
        self.id = id_
        self.n_channel = n_channel
        self.neighbors = neighbors
        self.lr = learning_rate
        self.gamma = reward_decay
        self.belief_decay = belief_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        # self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize the belief value
        self.chan_beliefs = np.zeros(self.n_channel)

        # initialize log data
        self.failure = 0
        self.success = 0
        self.n_chan_chosen = np.zeros(self.n_channel)

        # initialize replay memory [b, a, r, b_]
        self.memory = np.zeros((self.memory_size, n_channel * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params' + str(self.id))
        e_params = tf.get_collection('eval_net_params' + str(self.id))
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_history = []

    def _build_net(self):
        self.cell = tf.nn.rnn_cell.GRUCell(num_units=self.n_channel)


        # ----------------------- build evaluate_net used for action selection ----------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_channel], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_channel], name='Q_target')
        with tf.variable_scope('eval_net' + str(self.id)):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params' + str(self.id), tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_channel, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_channel], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_channel], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss' + str(self.id)):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train' + str(self.id)):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ----------------------- build target_net used for calculating target-Q value --------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_channel], name='s_')  # input
        with tf.variable_scope('target_net' + str(self.id)):
            c_names = ['target_net_params' + str(self.id), tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_channel, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_channel], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('l2', [1, self.n_channel], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, observation, reward, action):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # update the statistic
        if reward < 0:
            self.failure += 1
        else:
            self.success += 1
        self.n_chan_chosen[action] += 1

        b = self.chan_beliefs
        # update the channel belief
        for i in range(self.n_channel):
            if self.chan_beliefs[i] > 0.5:
                self.chan_beliefs[i] = max(0.5, self.belief_decay * self.chan_beliefs[i])
            else:
                self.chan_beliefs[i] = min(0.5, 1 - self.belief_decay * (1 - self.chan_beliefs[i]))
        self.chan_beliefs[action] = observation
        b_ = self.chan_beliefs

        transition = np.hstack((b, [action, reward], b_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self):
        # to have batch dimension when feed into tf placeholder
        observation = self.chan_beliefs[np.newaxis, :]

        # choose action with epsilon-greedy policy
        if np.random.uniform() < self.epsilon_max:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_channel)
        return action

    def learn(self):
        # check to replace the parameters of target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('successfully replace params in target net\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index,:]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_channel:],
                self.s: batch_memory[:, :self.n_channel],
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
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_channel],
                self.q_target: q_target
            }
        )
        self.cost_history.append(self.cost)
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('traning steps')
        plt.show()

    def stat(self):
        print('su%d: successful sensing ratio %f' % (self.id, self.success / (self.success + self.failure)))
        print(self.n_chan_chosen/np.sum(self.n_chan_chosen))


