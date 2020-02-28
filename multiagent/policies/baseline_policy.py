import numpy as np
from pyglet.window import key
from multiagent.policy import Policy

# interactive policy based on keyboard input
# hard-coded to deal only with movement, not communication
class BaselinePolicy(Policy):
    def __init__(self, env, agent_index):
        super(BaselinePolicy, self).__init__()
        self.env = env

        # hard-coded keyboard events
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]

        # Todo: Update with network

    def action(self, obs):

        # Todo: Update with network

        # ignore observation and just act based on keyboard events
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += .25
            if self.move[1]: u[2] += .25
            if self.move[3]: u[3] += .25
            if self.move[2]: u[4] += .25
            if True not in self.move:
                u[0] += 1.0
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
