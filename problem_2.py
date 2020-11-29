#################################################
#                                               #
# EL2805 Reinforcement Learning                 #
# Computer Lab 1                                #
# Problem 2                                     #
#                                               #
# Author: Tamas Faitli (19960205-T410)          #
#                                               #
#################################################

import numpy as np
import matplotlib.pyplot as plt
from MDP import MDP
from TableRenderer import TableRenderer

# Switches to run different parts of the exercise
# 0 - do not run
# 1 - run
RUN_CASES = {
    'solve_problem_and_render' : 1,
    'plot_value_as_discfactor' : 0
}

# Switch to decide whether plot or save images
# False - plot images
# True  - save images
SAVE_MODE = False

# Description of the map with the convention of:
# 0 : empty cell
# 1 : bank location
# 2 : police station
DEF_TABLE = np.array([
    [2,0,0,0,0,2],
    [0,0,3,0,0,0],
    [2,0,0,0,0,2]
])

# initial position
DEF_INIT_POS = (0,0,1,2)

# accuracy of the value iteration procedure
DEF_EPSILON = 0.0001

# render images
AGENT_IMG       = 'res/thief.npy'
MINOTAUR_IMG    = 'res/police.npy'

class BankRobbing(MDP):
    ''' BankRobbing class formulating the bank robbing problem as an MDP. '''

    # map constants
    BANK    = 2
    STATION = 3

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
    R_IDLE          = 0

    # number calculated on paper to check available police movement
    PARAM_MAX_DIST_INCREMENT = (2**(1/2)-1) + 0.05

    def __init__(self, table, init_pos):
        ''' Constructor..

        :param table:       np.array table describing the location
                            of banks and police station
        :param init_pos:    initial position (this problem is not episodic,
                            after the police catches the thief it transition
                            back to this position)
        '''
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
                # calculating expected reward for current state and action
                r = np.zeros((self.n_police_actions))
                p = np.zeros((self.n_police_actions))

                # possible state transitions and their reward
                for p_mov in range(self.n_police_actions):
                    s_next = self._MDP__move(s,a,p_mov)
                    p[p_mov] = self.transition_prob[s_next,s,a]
                    # catch
                    if self.states[s_next][0:2] == self.states[s_next][2:4]:
                        r[p_mov] = self.R_CATCH
                    # rob
                    elif self.states[s][0:2] == self.states[s_next][0:2] \
                        and self.table[self.states[s][0:2]] == self.BANK \
                        and a == self.STAY:
                        r[p_mov] = self.R_ROBBING
                    else:
                        r[p_mov] = self.R_IDLE

                # expected reward
                rewards[s,a] = np.dot(r,p)

        return rewards

    def __hitting_edge(self, pos):
        ''' Check if current position is outside of the table.

        :param pos: position (x,y)
        :return:    True if outside of table
                    False if on the table
        '''
        if (pos[0] <= -1) or (pos[0] >= self.table.shape[0]) \
            or (pos[1] <= -1) or (pos[1] >= self.table.shape[1]):
            return True
        else:
            return False

    def _MDP__move(self, state, action, police=None):
        ''' Evaluating current state and action and execute state
            transition.

        :param state:   Current state
        :param action:  Action taken
        :param police:  If None or left empty, police takes a random step.
                        If police action provided, it will take that step
                        in case that step is feasible (does not violate rules)
        :return:        Next state
        '''

        # parse position for agent and police from state
        agent_pos  = np.array(self.states[state][0:2])
        police_pos = np.array(self.states[state][2:4])

        # calculate distance
        orig_dist  = np.linalg.norm(agent_pos-police_pos)

        # catch, transition to initial state
        if orig_dist == 0.0:
            return self.map[self.init_pos]

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
                new_dist = np.linalg.norm(police_new_pos-agent_pos)
                if new_dist < (orig_dist+self.PARAM_MAX_DIST_INCREMENT):
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
        # there is no end condition, problem is not episodic
        return 0

    def _MDP__simulate_condition(self, flag, limit):
        ''' If the thief has been caught, it transitions back to init state.
            Therefore there is only time limitations.

        :param flag:
        :param limit:   Number of steps to simulate
        :return:        Boolean, True when hit count reached limit.
        '''
        if self.sim_time >= limit:
            return False
        else:
            self.sim_time += 1
            return True

    def __get_grid_values_for_fixed_pos(self, optimal_values, fixed_pos, default_val):
        ''' get values or policy for each agent position on the table for a
            fixed police value

        :param optimal_values:  Value or policy function (time invariant or for
                                one time step)
        :param fixed_pos:       The fixed position of the police
        :param default_val:
        :return:
        '''
        grid_val = np.zeros((self.table.shape))

        for r in range(self.table.shape[0]):
            for c in range(self.table.shape[1]):
                s = self.map[(r,c,fixed_pos[0],fixed_pos[1])]
                grid_val[r,c] = optimal_values[s]

        return grid_val

    def animate(self, renderer, path, policy=None, V=None, rewards=None, rate=0.3):
        ''' Animate the bank robbing problem for a given time.

        :param renderer:    TableRenderer object
        :param path:        List containing each state taken during the simulation
        :param policy:      Policy (if given) to plot optimal policy on the table
        :param V:           V (if given) to plot optimal values on the table
        :param rewards:     rewards (if given) prints accumulated rewards
        :param rate:        time while a single state is being plotted (in seconds)
        :return: -
        '''

        # time
        t = 0

        # iterate through path
        for step in path:
            # parse positions from path
            fixed_pos = step[2:4]

            if hasattr(policy, 'shape'):
                # time dependent policy
                if len(policy.shape) > 1:
                    # get drawable policy function
                    policy_t_fixed_pos = \
                        self.__get_grid_values_for_fixed_pos(policy[:, t], fixed_pos, self.STAY)
                # not time dependent
                else:
                    policy_t_fixed_pos = \
                        self.__get_grid_values_for_fixed_pos(policy[:], fixed_pos, self.STAY)
            else:
                policy_t_fixed_pos = None

            if hasattr(V, 'shape'):
                # time dependent value function
                if len(V.shape) > 1:
                    # get drawable value function
                    values_t_fixed_pos = \
                        self.__get_grid_values_for_fixed_pos(V[:, t], fixed_pos, 0.0)
                # not time dependent
                else:
                    values_t_fixed_pos = \
                        self.__get_grid_values_for_fixed_pos(V[:], fixed_pos, 0.0)
            else:
                values_t_fixed_pos = None

            # render state, values, policy
            renderer.update(step, policy_t_fixed_pos, values_t_fixed_pos, rate)

            if rewards != None:
                print("Cumulative reward: " + str(rewards[t]))

            # step
            t += 1

    def __police_actions(self):
        police = dict()
        police[self.RIGHT]  = ( 0, 1)
        police[self.UP]     = (-1, 0)
        police[self.LEFT]   = ( 0,-1)
        police[self.DOWN]   = ( 1, 0)
        return police

    def get_init_state(self):
        return self.map[self.init_pos]


def solve_mdp_and_animate_results(bank, renderer):

    V, policy = bank.solve_value_iteration(0.6, DEF_EPSILON)

    path, flag, rewards = banks.simulate(DEF_INIT_POS, policy, 20)

    banks.animate(renderer, path, policy, V)

def get_value_lambda_function(bank, lambdas):
    init_state = bank.get_init_state()

    values_at_init_state = []

    for l in lambdas:
        V, policy = bank.solve_value_iteration(l, DEF_EPSILON)
        values_at_init_state.append(V[init_state])

    return values_at_init_state


if __name__ == "__main__":
    banks = BankRobbing(DEF_TABLE, DEF_INIT_POS)

    character_images = {
        'agent'     : np.load(AGENT_IMG),
        'police'    : np.load(MINOTAUR_IMG)
    }
    renderer = TableRenderer(banks, character_images, SAVE_MODE, (6, 3))


    if RUN_CASES['solve_problem_and_render']:
        solve_mdp_and_animate_results(banks, renderer)

    if RUN_CASES['plot_value_as_discfactor']:
        lambdas = np.linspace(0.001,0.999,100)
        # lambdas[0] = 0.001
        values = get_value_lambda_function(banks, lambdas)

        plt.figure(figsize=(7,4))
        plt.plot(lambdas, values)
        plt.xlabel("Discount factor")
        plt.ylabel("Value function evaluated in initial state")
        plt.show()



