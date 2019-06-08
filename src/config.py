"""
This part of code is about ...
"""

import numpy as np


class Config:
    # agent network configuration
    n_su = 9
    n_pu = 8
    neighbors = [[1, 3], [0, 2, 4], [1, 5],
                  [0, 4, 6], [1, 3, 5, 7], [2, 4, 8],
                  [3, 7], [4, 6, 8], [5, 7]]


class ConfigAgent(Config):

    # q-network configuration
    learning_rate = 0.01
    reward_decay = 0.9
    e_greedy = 0.95
    replace_target_iter = 100

    # agent configuration
    belief_decay = 0.9
    memory_size = 20
    batch_size = 10


class ConfigEnv(Config):

    # plot configuration
    UNIT = 100  # pixels
    FIELD = 6  # grid size

    # SU configuration
    detection_threshold = 0.05
    n_detection_sample = 900

    # PU configuration
    pu_speed_max = 16
    pu_speed_min = 0
    pu_state = np.ones(Config.n_pu).astype(int)
    pu_state_trans_prob = np.random.uniform(size=(Config.n_pu, 2))  # (p(1|0), p(0|1))
    pu_pos = np.random.uniform(0, FIELD * UNIT, size=(Config.n_pu, 2))
    BUSY = 0
    IDLE = 1

    # radio environment configuration
    path_loss_exp = 2.8
    noise_var = 0.01

