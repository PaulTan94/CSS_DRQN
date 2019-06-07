"""
This part of code is about the configuration and the update of radio environment

    1. Network topology for Secondary User(su)
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
# if sys.version_info.major == 2:
#     import Tkinter as tk
# else:
import tkinter as tk
from scipy.stats import norm

UNIT = 100      # pixels
FIELD = 6   # grid size
BUSY = 0
IDLE = 1


class Radio(tk.Tk, object):
    def __init__(self):
        super(Radio, self).__init__()
        self.neighbors = [[1, 3], [0, 2, 4], [1, 5],
                          [0, 4, 6], [1, 3, 5, 7], [2, 4, 8],
                          [3, 7], [4, 6, 8], [5, 7]]
        self.n_su = 9
        self.n_pu = 8
        self.pu_speed_max = 16
        self.pu_speed_min = 8
        self.title("Radio Environment")
        self.geometry('{0}x{1}'.format(FIELD * UNIT, FIELD * UNIT))
        self.pu_state = np.ones(self.n_pu).astype(int)
        self.pu_state_trans_prob = np.random.uniform(size=(self.n_pu, 2))  # (p(1|0), p(0|1))
        self.pu_pos = np.random.uniform(0, FIELD * UNIT, size=(self.n_pu, 2))
        self.path_loss_exp = 2
        self.detection_threshold = 0.05
        self.n_detection_sample = 900
        self.noise_var = 0.01
        self._build_radio()

    def _build_radio(self):
        self.canvas = tk.Canvas(self, bg='white', height=FIELD * UNIT, width=FIELD * UNIT)

        # create grids
        for c in range(0, FIELD, 1):
            self.canvas.create_line(c * UNIT, 0, c * UNIT, FIELD * UNIT)
        for r in range(0, FIELD, 1):
            self.canvas.create_line(0, r * UNIT, FIELD * UNIT, r * UNIT)

        # create SUs
        self.su = []
        for i in range(self.n_su):
            pos_x = (2 * (i % 3) + 1) * UNIT
            pos_y = (2 * (i // 3) + 1) * UNIT
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
        self.timestep = self.canvas.create_text(50, FIELD * UNIT - 20, text='Time step: 0')

        # pack all
        self.canvas.pack()

    def reset(self):
        pass

    def observe(self, action, su_index):
        # receives signal and obtains observation (sensing decision)
        if self.pu_state[action] == BUSY:  # busy channel
            coord_diff = np.array(self.canvas.coords(self.pu[action])) - np.array(self.canvas.coords(self.su[su_index]))
            dist = np.linalg.norm(coord_diff) / (2 * UNIT)
            signal_strg = min(1, np.power(dist, - self.path_loss_exp)) + self.noise_var
            z_value = np.sqrt(self.n_detection_sample / 2) * (self.detection_threshold / signal_strg - 1)
            detection_prob = 1 - norm.cdf(z_value)
            if np.random.uniform() < detection_prob:
                observation = BUSY   # true negative
                reward = -1
            else:
                observation = IDLE   # false positive
                reward = -1.5
        else:  # idle channel
            signal_strg = self.noise_var
            z_value = np.sqrt(self.n_detection_sample / 2) * (self.detection_threshold / signal_strg - 1)
            false_alarm_prob = 1 - norm.cdf(z_value)
            if np.random.uniform() < false_alarm_prob:
                observation = BUSY   # false negative
                reward = -1
            else:
                observation = IDLE   # true positive
                reward = 1

        if observation != self.pu_state[action]:
            self.canvas.itemconfig(self.su[su_index], fill="yellow")
        else:
            self.canvas.itemconfig(self.su[su_index], fill="blue")

        # return signal strength
        return observation, reward

    def step(self, timestep):
        for i in range(self.n_pu):
            # next pu state
            state = self.pu_state[i]
            if np.random.uniform() > self.pu_state_trans_prob[i][state]:
                self.pu_state[i] = 1 - self.pu_state[i]
            if self.pu_state[i] == 0:
                self.canvas.itemconfig(self.pu[i], fill="red")
            else:
                self.canvas.itemconfig(self.pu[i], fill="green")

            # update pu position using random walk model with reflection
            angle = np.random.uniform() * 2 * np.pi
            speed = np.random.uniform(self.pu_speed_min, self.pu_speed_max)
            delta_x, delta_y = speed * np.cos(angle), speed * np.sin(angle)
            pos = self  .canvas.coords(self.pu[i])
            if pos[0] + delta_x < 0:
                delta_x = - 2 * pos[0] - delta_x
            elif pos[0] + delta_x > FIELD * UNIT:
                delta_x = 2 * FIELD * UNIT - 2 * pos[0] - delta_x

            if pos[1] + delta_y < 0:
                delta_y = - 2 * pos[1] - delta_y
            elif pos[1] + delta_y > FIELD * UNIT:
                delta_y = 2 * FIELD * UNIT - 2 * pos[1] - delta_y

            self.canvas.move(self.pu[i], delta_x, delta_y)
            self.canvas.itemconfig(self.timestep, text='Time step: ' + str(timestep))

    def render(self):
        self.update()

    def stat(self):
        chan_vc_prob = self.pu_state_trans_prob[:, 0] / np.sum(self.pu_state_trans_prob, axis=1)
        print('Channel vacant probability:')
        print(chan_vc_prob)
        print('Average vacant probability: %f' % np.mean(chan_vc_prob))




