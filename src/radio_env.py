"""
This part of code is about the configuration and the update of radio environment

    1. Network topology for Secondary User(SU)
        |0 1 2|
        | 3 4 5|
        | 6 7 8|
    2. Primary User(PU) state transition matrix
        | p(1|0) p(0|0) |
        | p(0|1) p(1|1) |
        0 for busy, 1 for idle
    3.
"""

import numpy as np
import tkinter as tk
from scipy.stats import norm


class Radio(tk.Tk, object):
    def __init__(self, config):
        super(Radio, self).__init__()

        # plot configuration
        self.UNIT = config.UNIT
        self.FIELD = config.FIELD

        # SU configuration
        self.neighbors = config.neighbors
        self.n_su = config.n_su
        self.detection_threshold = config.detection_threshold
        self.n_detection_sample = config.n_detection_sample

        # PU configuration
        self.n_pu = config.n_pu
        self.pu_speed_max = config.pu_speed_max
        self.pu_speed_min = config.pu_speed_min
        self.pu_state = config.pu_state
        self.pu_state_trans_prob = config.pu_state_trans_prob
        self.pu_pos = config.pu_pos
        self.BUSY = config.BUSY
        self.IDLE = config.IDLE

        # radio environment configuration
        self.path_loss_exp = config.path_loss_exp
        self.noise_var = config.noise_var

        self.title("Radio Environment")
        self.geometry('{0}x{1}'.format(self.FIELD * self.UNIT, self.FIELD * self.UNIT))
        self._build_radio()

    def _build_radio(self):
        self.canvas = tk.Canvas(self, bg='white', height=self.FIELD * self.UNIT, width=self.FIELD * self.UNIT)

        # create grids
        for c in range(0, self.FIELD, 1):
            self.canvas.create_line(c * self.UNIT, 0, c * self.UNIT, self.FIELD * self.UNIT)
        for r in range(0, self.FIELD, 1):
            self.canvas.create_line(0, r * self.UNIT, self.FIELD * self.UNIT, r * self.UNIT)

        # create SUs
        self.su = []
        for i in range(self.n_su):
            pos_x = (2 * (i % 3) + 1) * self.UNIT
            pos_y = (2 * (i // 3) + 1) * self.UNIT
            self.su.append(self.canvas.create_rectangle(
                 pos_x - 10, pos_y - 10,
                 pos_x + 10, pos_y + 10, fill='blue'
            ))

        # create PUs
        self.pu = []
        for i in range(self.n_pu):
            self.pu.append(self.canvas.create_oval(
                self.pu_pos[i][0] - 10, self.pu_pos[i][1] - 10,
                self.pu_pos[i][0] + 10, self.pu_pos[i][1] + 10, fill='green'
            ))

        # create text
        self.txt_timestep = self.canvas.create_text(50, self.FIELD * self.UNIT - 20, text='Time step: 0')

        # pack all
        self.canvas.pack()

    def observe(self, action, su_index):
        # receives signal and obtains observation (sensing decision)
        if self.pu_state[action] == self.BUSY:  # busy channel
            coord_diff = np.array(self.canvas.coords(self.pu[action])) - np.array(self.canvas.coords(self.su[su_index]))
            dist = np.linalg.norm(coord_diff) / (2 * self.UNIT)
            signal_strg = min(1, np.power(dist, - self.path_loss_exp)) + self.noise_var
            z_value = np.sqrt(self.n_detection_sample / 2) * (self.detection_threshold / signal_strg - 1)
            detection_prob = 1 - norm.cdf(z_value)
            if np.random.uniform() < detection_prob:
                observation = self.BUSY   # true negative
                reward = -1
            else:
                observation = self.IDLE   # false positive
                reward = -1.5
        else:  # idle channel
            signal_strg = self.noise_var
            z_value = np.sqrt(self.n_detection_sample / 2) * (self.detection_threshold / signal_strg - 1)
            false_alarm_prob = 1 - norm.cdf(z_value)
            if np.random.uniform() < false_alarm_prob:
                observation = BUSY   # false negative
                reward = -1
            else:
                observation = self.IDLE   # true positive
                reward = 1

        if observation != self.pu_state[action]:
            self.canvas.itemconfig(self.su[su_index], fill="yellow")
        else:
            self.canvas.itemconfig(self.su[su_index], fill="blue")

        # return signal strength
        return observation, reward

    def step(self, timestep):
        for i in range(self.n_pu):
            # update pu state
            state = self.pu_state[i]
            if np.random.uniform() > self.pu_state_trans_prob[i][state]:
                self.pu_state[i] = 1 - self.pu_state[i]
            if self.pu_state[i] == 0:
                self.canvas.itemconfig(self.pu[i], fill="red")
            else:
                self.canvas.itemconfig(self.pu[i], fill="green")

            # update pu position
            angle = np.random.uniform() * 2 * np.pi  # random walk model with reflection
            speed = np.random.uniform(self.pu_speed_min, self.pu_speed_max)
            delta_x, delta_y = speed * np.cos(angle), speed * np.sin(angle)
            pos = self  .canvas.coords(self.pu[i])
            if pos[0] + delta_x < 0:
                delta_x = - 2 * pos[0] - delta_x
            elif pos[0] + delta_x > self.FIELD * self.UNIT:
                delta_x = 2 * self.FIELD * self.UNIT - 2 * pos[0] - delta_x

            if pos[1] + delta_y < 0:
                delta_y = - 2 * pos[1] - delta_y
            elif pos[1] + delta_y > self.FIELD * self.UNIT:
                delta_y = 2 * self.FIELD * self.UNIT - 2 * pos[1] - delta_y

            # update canvas
            self.canvas.move(self.pu[i], delta_x, delta_y)
            self.canvas.itemconfig(self.txt_timestep, text='Time step: ' + str(timestep))

    def render(self):
        self.update()

    def stat(self):
        chan_vc_prob = self.pu_state_trans_prob[:, 0] / np.sum(self.pu_state_trans_prob, axis=1)
        print('Channel vacant probability:')
        print(chan_vc_prob)
        print('Average vacant probability: %f' % np.mean(chan_vc_prob))




