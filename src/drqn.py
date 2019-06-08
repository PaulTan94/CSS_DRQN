"""
This part of code is about ...
"""

# import numpy as np
import tensorflow as tf

# np.random.seed(1)
tf.set_random_seed(1)


# DRQN Network off-policy
class DRQN:
    def __init__(self, _id, config):
        self.su_id = _id
        self.n_channel = config.n_pu
        self.lr = config.learning_rate

    def add_placeholders(self):
        self.s = tf.placeholder(tf.float32, [None, self.n_channel], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_channel], name='Q_target')

        self.s_ = tf.placeholder(tf.float32, [None, self.n_channel], name='s_')  # input

    def add_eval_net(self):
        # ----------------------- build evaluate_net used for action selection ----------------------
        with tf.variable_scope('eval_net' + str(self.su_id)):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params' + str(self.su_id), tf.GraphKeys.GLOBAL_VARIABLES], 10, \
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

    def add_target_net(self):
        # ----------------------- build target_net used for calculating target-Q value ---------------
        with tf.variable_scope('target_net' + str(self.su_id)):
            c_names, n_l1, w_initializer, b_initializer = \
                ['target_net_params' + str(self.su_id), tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

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

    def train(self):
        with tf.variable_scope('loss' + str(self.su_id)):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train' + str(self.su_id)):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def replace_target(self):
        # consist of [target_net, evaluate_net]
        t_params = tf.get_collection('target_net_params' + str(self.su_id))
        e_params = tf.get_collection('eval_net_params' + str(self.su_id))
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def build(self):
        self.replace_target()
        self.add_placeholders()
        self.add_eval_net()
        self.add_target_net()
        self.train()
