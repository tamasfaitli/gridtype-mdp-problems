#################################################
#                                               #
# EL2805 Reinforcement Learning                 #
# Computer Lab 1                                #
# Problem 1                                     #
#                                               #
# Author: Tamas Faitli (19960205-T410)          #
# Part of the code is inherited from lab0       #
#                                               #
#################################################

import numpy as np
import matplotlib.pyplot as plt
from TableRenderer import TableRenderer
from MDP import MDP

# Description of the maze
# with the convention of
# 0 = empty cell
# 1 = wall
# 2 = exit
DEF_MAZE = np.array([[0,0,1,0,0,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,1,1],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,1,1,1,1,1,1,0],
                    [0,0,0,0,1,2,0,0]])

# render images
AGENT_IMG       = 'res/agent_comp.npy'
MINOTAUR_IMG    = 'res/minotaur_comp.npy'

class Maze(MDP):
    # maze constants
    WALL    = 1
    EXIT    = 2

    # situation
    CONT    = 0
    WIN     = 1
    DEAD    = -1

    # actions
    RIGHT   = 0
    UP      = 1
    LEFT    = 2
    DOWN    = 3
    STAY    = 4

    # rewards
    R_STEP          = 0
    R_GOAL          = 1
    R_IMPOSSIBLE    = -100
    R_DIE           = -50

    def __init__(self, table, minotaur=False):
        '''

        :param table:
        :param minotaur: True if it can stay, False if it always has to move
        '''
        self.table = table
        self.minotaur_mov               = self.__minotaur_movement(minotaur)
        self.n_minotaur_mov             = len(self.minotaur_mov)
        super().__init__()


    def _MDP__actions(self):
        actions = dict()
        actions[self.RIGHT] = (0, 1)
        actions[self.UP]    = (-1,0)
        actions[self.LEFT]  = (0,-1)
        actions[self.DOWN]  = (1, 0)
        actions[self.STAY]  = (0, 0)
        return actions

    def _MDP__states(self):
        states = dict()
        map = dict()
        s = 0
        # agent position
        for row in range(self.table.shape[0]):
            for col in range(self.table.shape[1]):
                if self.table[row, col] != self.WALL:
                    # minotaur position
                    # with the assumption it can walk through wall
                    # but cannot not stay inside
                    for m_row in range(self.table.shape[0]):
                        for m_col in range(self.table.shape[1]):
                            if self.table[m_row, m_col] != self.WALL:
                                states[s] = (row,col,m_row,m_col)
                                map[(row,col,m_row,m_col)] = s
                                s += 1

        return states, map

    def _MDP__move(self, state, action, det_minotaur_movement=None):
        '''

        :param state:
        :param action:
        :param offline: switch to use this function
        :return:
        '''
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]

        # agent hits wall or edge
        if (row == -1) or (row == self.table.shape[0]) or \
            (col == -1) or (col == self.table.shape[1]) or \
            (self.table[row, col] == self.WALL):

            # stay in current position
            row = self.states[state][0]
            col = self.states[state][1]

        # current position of minotaur
        m_row = self.states[state][2]
        m_col = self.states[state][3]

        # check available movements of the minotaur
        minotaur_next_positions = {}
        for m in range(self.n_minotaur_mov):
            valid, pos = self.__step_minotaur([m_row,m_col], self.minotaur_mov[m])
            if valid:
                minotaur_next_positions[m] = pos

        # deterministic minotaur movement for transition and reward filling
        if det_minotaur_movement != None:
            # check if this movement is feasible
            if det_minotaur_movement in minotaur_next_positions:
                minotaur_next_position = minotaur_next_positions[det_minotaur_movement]
            else:
                minotaur_next_position = [m_row, m_col]
        # draw a next movement using uniform distribution
        else:
            minotaur_step = np.random.choice(list(minotaur_next_positions.keys()))
            minotaur_next_position = minotaur_next_positions[minotaur_step]

        return self.map[(row,col,minotaur_next_position[0],minotaur_next_position[1])]


    def _MDP__transition_probabilities(self):
        ''' Defining transition probabilities for each state transition for each action

        :return: transitions: p(s'|s,a)
        '''
        dim = (self.n_states, self.n_states, self.n_actions)
        transitions = np.zeros(dim)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = []
                for m in range(self.n_minotaur_mov):
                    next_s = self._MDP__move(s,a,m)
                    if next_s not in next_states:
                        next_states.append(next_s)
                for next_s in next_states:
                    transitions[next_s, s, a] = 1/len(next_states)

        return transitions

    def _MDP__rewards(self):
        ''' Rewards at each state for each action

        :return: rewards: r(s,a)
        '''
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                for m_mov in range(self.n_minotaur_mov):
                    s_next = self._MDP__move(s,a,m_mov)
                    # agent hits the wall
                    if self.states[s][0:2] == self.states[s_next][0:2] and a != self.STAY:
                        rewards[s,a] = self.R_IMPOSSIBLE
                    # minotaur catches the agent
                    elif self.states[s_next][0:2] == self.states[s_next][2:4]:
                        rewards[s,a] = self.R_DIE
                    # reward for reaching exit
                    elif self.states[s][0:2] == self.states[s_next][0:2] \
                            and self.table[self.states[s_next][0:2]] == self.EXIT \
                            and a == self.STAY:
                    # elif self.maze[self.states[s_next][0:2]] == self.EXIT \
                    #         and a != self.STAY:
                        rewards[s,a] = self.R_GOAL
                    else:
                        rewards[s,a] = self.R_STEP

        return rewards

    def _MDP__end_condition(self, s, next_s):
        ''' Determine whether the realized state transition ends the game

        :param s:       Current state
        :param next_s:  Next state
        :return: flag:  self.DIE if minotaur catches the agent
                        self.WIN if the agent exits the maze
                        self.CONT if nothing happened
        '''
        # at the step taken the minotaur catches the agent
        position = self.states[next_s]
        if position[0:2] == position[2:4]:
            return self.DEAD
        # at the step taken the agent exited the maze
        elif self.table[position[0:2]] == self.EXIT:
            return self.WIN
        else:
            return self.CONT


    def _MDP__simulate_condition(self, flag, limit=None):
        ''' Defining the condition to stop the simulation

        :param flag:    values defined in method __end_condition()
        :param limit:   None: Not used for this maze. It always ends with either
                        exiting or dying.
                        Int:  Maximum number of steps the agent has to exit.
        :return:        Boolean whether to continue the simulation. (True to continue)
        '''
        to_continue = True

        if limit != None:
            if self.sim_time >= limit:
                to_continue = False
            else:
                self.sim_time += 1

        if flag != self.CONT:
            to_continue = False

        return to_continue

    def __minotaur_movement(self, can_stay):
        '''

        :param can_stay: True if it can stay, False if not
        :return:
        '''
        minotaur = dict()
        minotaur[self.RIGHT] = (0, 1)
        minotaur[self.UP]    = (-1,0)
        minotaur[self.LEFT]  = (0,-1)
        minotaur[self.DOWN]  = (1, 0)
        if can_stay:
            minotaur[self.STAY]  = (0, 0)
        return minotaur

    def __step_minotaur(self, pos, step):
        # step
        r = pos[0] + step[0]
        c = pos[1] + step[1]

        # hitting the edge:
        edge = (r == -1) or (r == self.table.shape[0]) or \
               (c == -1) or (c == self.table.shape[1])
        if edge:
            return False, [0,0]

        # going through wall
        elif self.table[r, c] == self.WALL:
            valid, pos = self.__step_minotaur([r,c],step)
            if not valid:
                return False, [0,0]
            else:
                return True, pos

        # feasible movement
        return True, [r,c]

    def get_grid_values_for_minpos(self, optimal_values, minotaur_pos, default_val):
        grid_val = np.zeros((self.table.shape))

        for r in range(self.table.shape[0]):
            for c in range(self.table.shape[1]):
                # wall positions are not states
                if self.table[r, c] != 1:
                    s = self.map[(r,c,minotaur_pos[0],minotaur_pos[1])]
                    grid_val[r,c] = optimal_values[s]
                else:
                    grid_val[r,c] = default_val

        return grid_val


    def animate(self, renderer, path, policy=None, V=None):
        # time
        t = 0

        # iterate through path
        for step in path:
            # parse positions from path
            agent_pos = step[0:2]
            minotaur_pos = step[2:4]

            if hasattr(policy, 'shape'):
                # time dependent policy
                if len(policy.shape) > 1:
                    # get drawable policy function
                    policy_t_minpos = self.get_grid_values_for_minpos(policy[:, t], minotaur_pos, self.STAY)
                # not time dependent
                else:
                    policy_t_minpos = self.get_grid_values_for_minpos(policy[:], minotaur_pos, self.STAY)
            else:
                policy_t_minpos = None

            if hasattr(V, 'shape'):
                # time dependent value function
                if len(V.shape) > 1:
                    # get drawable value function
                    values_t_minpos = self.get_grid_values_for_minpos(V[:, t], minotaur_pos, 0.0)
                # not time dependent
                else:
                    values_t_minpos = self.get_grid_values_for_minpos(V[:], minotaur_pos, 0.0)
            else:
                values_t_minpos = None

            # render state, values, policy
            renderer.update(step, policy_t_minpos, values_t_minpos, 0.3)

            # step
            t += 1

    def get_win_flag(self):
        return self.WIN



def problem_b_dynprog(maze, renderer):
    ''' Solve the problem, and illustrate an optimal policy for T = 20.

    :param maze:
    :param renderer:
    :return:
    '''
    T = 20
    start = (0, 0, 6, 5)

    V, policy = maze.solve_dynamic_programming(T)

    path, win = maze.simulate(start, policy)

    maze.animate(renderer, path, policy, V)

    n_games = 10000
    n_win = 0

    for game in range(n_games):
        path, win = maze.simulate(start, policy)
        if win == maze.get_win_flag():
            n_win += 1

    print(n_win/n_games)


def problem_b_valueiter(maze, renderer):
    ''' Solve the problem, and illustrate an optimal policy for T = 20.

    :param maze:
    :param renderer:
    :return:
    '''
    gamma = 0.95
    epsilon = 0.0001
    start = (0, 0, 6, 5)

    V, policy = maze.solve_value_iteration(gamma, epsilon)

    path, win = maze.simulate(start, policy)

    maze.animate(renderer, path, policy, V)

    return policy


def plot_win_prob_time(time, win_rate, aux=None):
    plt.figure(figsize=(7,4))
    plt.plot(time, win_rate, 'yD')
    plt.plot(time, win_rate, 'k:', label='_nolegend_')
    if hasattr(aux, 'shape'):
        plt.plot(time, aux, 'mD')
        plt.plot(time, aux, 'k:', label='_nolegend_')
        plt.legend(["Minotaur always move","Minotaur can stay still"])
    plt.xticks(time)
    plt.xlabel("T - available time to exit the maze")
    plt.ylabel("p(exit|T)")

    plt.show()

def problem_b_prob_of_exiting_dynprog(maze):
    start = (0, 0, 6, 5)

    horizons = np.arange(20) + 1
    win_prob = np.zeros(horizons.shape)

    n_games = 10000

    for t in horizons:
        print("Solving maze using dynamic programming with horizon: " + str(t) + "...")
        V, policy = maze.solve_dynamic_programming(t)
        print("Maze solved...")

        print("Estimating win rate for time: " + str(t) + "...")
        n_win = 0

        for game in range(n_games):
            path, win = maze.simulate(start, policy, t)
            if win == maze.get_win_flag():
                n_win +=1

        win_prob[t-1] = n_win/n_games
        print("Winning probability: " + str(win_prob[t-1]) + "...")


    return horizons, win_prob

def problem_b_prob_of_exiting(maze, policy=None):
    ''' Plotting the maximal probability of exiting the maze as a function of T.

    :param maze:
    :param policy:
    :return:
    '''
    gamma   = 0.95
    epsilon = 0.0001
    start   = (0, 0, 6, 5)

    if not hasattr(policy, 'shape'):
        print("Solving maze...")
        V, policy = maze.solve_value_iteration(gamma, epsilon)
        print("Maze has been solved...")

    time        = np.arange(20) + 1
    win_prob    = np.zeros(time.shape)

    n_games = 10000

    for t in time:
        print("Estimating win rate for time: " + str(t) + "...")
        n_win = 0

        for game in range(n_games):
            path, win = maze.simulate(start, policy, t)
            if win == maze.get_win_flag():
                n_win +=1

        win_prob[t-1] = n_win/n_games
        print("Winning probability: " + str(win_prob[t-1]) + "...")


    return time, win_prob


def problem_c():
    pass


if __name__ == '__main__':
    # False to plot, true to save images
    save_mode = False

    # init maze
    maze = Maze(DEF_MAZE)

    # init renderer
    character_images = {
        'agent'     : np.load(AGENT_IMG),
        'minotaur'  : np.load(MINOTAUR_IMG)}
    renderer = TableRenderer(maze, character_images, save_mode)

    # running code for (b) using dynamic programming
    problem_b_dynprog(maze, renderer)

    # running code for (b) using value iteration
    # policy = None
    # policy = problem_b_valueiter(maze, renderer)

    # plotting winning rate as function of T
    # problem_b_prob_of_exiting(maze, policy)

    # plotting winning rate as function of T solving each time with dynprog
    # time, winrate = problem_b_prob_of_exiting_dynprog(maze)
    # maze = Maze(DEF_MAZE, True)
    # time_minstay, winrate_minstay = problem_b_prob_of_exiting_dynprog(maze)
    # plot_win_prob_time(time, winrate, winrate_minstay)

    # running code for (c)
    problem_c()



