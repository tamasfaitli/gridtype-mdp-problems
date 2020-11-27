import numpy as np
import matplotlib.pyplot as plt
from MDP import MDP

# Description of the map with the convention of:
# 0 : empty cell
# 1 : bank location
# 2 : police station
DEF_TABLE = np.array([
    [1,0,0,0,0,1],
    [0,0,2,0,0,0],
    [1,0,0,0,0,1]
])

DEF_INIT_POS = (0,0,1,2)

class BankRobbing(MDP):
    # map constants
    BANK    = 1
    STATION = 2

    # situation
    CATCH   = -1
    CONT    = 0

    # actions
    RIGHT   = 0
    UP      = 1
    LEFT    = 2
    DOWN    = 3
    STAY    = 4

    # rewards
    R_ROBBING       = 10
    R_CATCH         = -50
    R_IMPOSSIBLE    = -100

    # number calculated on paper to check available police movement
    PARAM_MAX_DIST_INCREMENT = 0.236 + 0.05

    def __init__(self, table, init_pos):
        self.table = table
        self.police_actions     = self.__police_actions()
        self.n_police_actions   = len(self.police_actions)
        self.init_pos           = init_pos
        super().__init__()

    def _MDP__states(self):
        states  = dict()
        map     = dict()
        s       = 0

        for ar in range(self.table.shape[0]):
            for ac in range(self.table.shape[1]):
                for pr in range(self.table.shape[0]):
                    for pc in range(self.table.shape[1]):
                        states[s] = (ar,ac,pr,pc)
                        map[(ar,ac,pr,pc)] = s
                        s += 1

        return states, map

    def _MDP__actions(self):
        actions = dict()
        actions[self.RIGHT] = (0, 1)
        actions[self.UP]    = (-1,0)
        actions[self.LEFT]  = (0,-1)
        actions[self.DOWN]  = (1, 0)
        actions[self.STAY]  = (0, 0)
        return actions

    def _MDP__transition_probabilities(self):
        dim = (self.n_states, self.n_states, self.n_actions)
        transitions = np.zeros(dim)

        for s in range(self.n_states):
            # in case of catch, always transition to initial position
            if self.states[s][0:2] == self.states[s][2:4]:
                for a in range(self.n_actions):
                    transitions[self.map[self.init_pos], s, a] = 1
            else:
                for a in range(self.n_actions):
                    next_states = []
                    for p in range(self.n_police_actions):
                        # check how many possible actions can the
                        # police make at current state
                        next_s = self._MDP__move(s,a,p)
                        if (self.states[s][2:4] != self.states[next_s][2:4]) \
                            and (next_s not in next_states):
                            next_states.append(next_s)
                    for next_s in next_states:
                        transitions[next_s, s, a] = 1/len(next_states)

        return transitions

    def _MDP__rewards(self):
        ''' Defining reward function...
             - Assumption the agent shall STAY at the bank to rob it
        :return: rewards : reward for each state-action pair
        '''
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                for p in range(self.n_police_actions):
                    next_s = self._MDP__move(s,a,p)
                    # catch
                    if self.states[next_s][0:2] == self.states[next_s][2:4]:
                        rewards[s,a] = self.R_CATCH
                    # rob
                    if self.states[s][0:2] == self.states[next_s][0:2] \
                        and a == self.STAY:
                        rewards[s,a] = self.R_ROBBING

        return rewards

    def __hitting_edge(self, pos):
        if (pos[0] == -1) or (pos[0] == self.table.shape[0]) \
            or (pos[1] == -1) or (pos[1] == self.table.shape[1]):
            return True
        else:
            return False

    def _MDP__move(self, state, action, police=None):
        agent_pos  = np.array(self.states[state][0:2])
        police_pos = np.array(self.states[state][2:4])

        orig_dist  = np.linalg.norm(agent_pos-police_pos)

        act = np.array(self.actions[action])

        # update agent pos
        agent_new_pos = agent_pos + act

        # agent hits the edge
        if self.__hitting_edge(agent_new_pos):
            # stay in current position
            agent_new_pos = agent_pos

        # check available police movements
        police_next_positions = {}
        for p in range(self.n_police_actions):
            police_act = np.array(self.police_actions[p])
            police_new_pos = police_pos + police_act

            # check whether new position is not on the table
            if not self.__hitting_edge(police_new_pos):
                diff = np.linalg.norm(police_new_pos-agent_pos)
                if diff < self.PARAM_MAX_DIST_INCREMENT:
                    police_next_positions[p] = police_new_pos

        # deterministic police movement for transition and reward filling
        if police != None:
            if police in police_next_positions:
                police_new_pos = police_next_positions[police]
            else:
                police_new_pos = police_pos
        # draw a random next movement using uniform distribution
        else:
            police_step = np.random.choice(list(police_next_positions.keys()))
            police_new_pos = police_next_positions[police_step]

        return self.map[(agent_new_pos[0],agent_new_pos[1],police_new_pos[0],police_new_pos[1])]

    def _MDP__end_condition(self, s, next_s):
        pass

    def _MDP__simulate_condition(self, flag, limit=None):
        pass

    def __police_actions(self):
        police = dict()
        police[self.RIGHT]  = ( 0, 1)
        police[self.UP]     = (-1, 0)
        police[self.LEFT]   = ( 0,-1)
        police[self.DOWN]   = ( 1, 0)
        return police

if __name__ == "__main__":

    banks = BankRobbing(DEF_TABLE, DEF_INIT_POS)

    V, policy = banks.solve_value_iteration(0.95, 0.0001)

    pass

