import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(3)]
        for i, agent in enumerate(world.agents):
            if(i==3):
                agent.name = 'agent %d' % i
                agent.adversary = True
                agent.name = 'agent %d' % i
                agent.collide = False
                agent.size = 0.075
            else:
                agent.name = 'agent %d' % i
                agent.adversary = False
                agent.collide = False
                agent.size = 0.075
        # speaker
        world.agents[0].movable = False
        # listener
        world.agents[1].silent = True
        # adversary
        world.agents[2].silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.04
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        # want listener to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        mygoal = np.random.choice(world.landmarks)
        world.agents[0].goal_b = mygoal
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])               
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])
        # special colors for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color + np.array([0.45, 0.45, 0.45])

        world.agents[0].key = np.random.choice(world.landmarks).color

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return self.reward(agent, reward)

    def reward(self, agent, world):
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # squared distance from listener to landmark
        good_speaker = world.agents[0]
        adverse_listener = world.agents[2]
        dist2 = np.sum(np.square(good_speaker.goal_a.state.p_pos - good_speaker.goal_b.state.p_pos))
        adv_dist2 = np.sum(np.square(adverse_listener.state.p_pos - good_speaker.goal_b.state.p_pos))
        return adv_dist2 - dist2

    def adversary_reward(self, agent, world):
        # squared distance from listener to landmark
        good_speaker = world.agents[0]
        adverse_listener = world.agents[2]
        dist2 = np.sum(np.square(good_speaker.goal_a.state.p_pos - good_speaker.goal_b.state.p_pos))
        adv_dist2 = np.sum(np.square(adverse_listener.state.p_pos - good_speaker.goal_b.state.p_pos))
        return dist2 - adv_dist2

    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None): continue
            comm.append(other.state.c)
        
        if world.agents[0].key is None:
            confer = np.array([1])
            key = np.zeros(world.dim_c)
            goal_color = np.zeros(world.dim_c)
        else:
            key = world.agents[0].key


        # speaker
        if not agent.movable:
            return np.concatenate([goal_color])
        # listener
        if agent.silent and not agent.adversary:
            return np.concatenate([agent.state.p_vel] + entity_pos + [key] + comm)
        # adversary
        if agent.silent:
            return np.concatenate([agent.state.p_vel] + entity_pos + comm)
                  
