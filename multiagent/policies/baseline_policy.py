import numpy as np
from pyglet.window import key
from multiagent.policy import Policy
import torch.nn as nn
import torch.nn.functional as F

# hard-coded to deal only with movement, not communication
class BaselinePolicy(Policy):
    def __init__(self, env, agent_index):
        super(BaselinePolicy, self).__init__()
        self.env = env

        # Network
        print(self.env.action_space[0].shape,*self.env.observation_space[0].shape)
        self.linear1 = nn.Linear(*self.env.observation_space[0].shape, 64)
        self.linear2 = nn.Linear(64, *self.env.action_space[0].shape)

    def action(self, obs):

        # Todo: Update with network

        # ignore observation and just act based on keyboard events
        # if self.env.discrete_action_input:
        #     u = 0
        #     if self.move[0]: u = 1
        #     if self.move[1]: u = 2
        #     if self.move[2]: u = 4
        #     if self.move[3]: u = 3
        # else:
        #     u = np.zeros(5) # 5-d because of no-move action
        #     if self.move[0]: u[1] += .25
        #     if self.move[1]: u[2] += .25
        #     if self.move[3]: u[3] += .25
        #     if self.move[2]: u[4] += .25
        #     if True not in self.move:
        #         u[0] += 1.0
        print("obs",obs)
        out = self.linear1(torch.FloatTensor(obs)).clamp(min=0)
        out = F.sigmoid(self.linear2(out))
        return np.concatenate([out.data.nump(), np.zeros(self.env.world.dim_c)])
