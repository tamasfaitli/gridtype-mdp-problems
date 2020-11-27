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
from MazeRenderer import MazeRenderer

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

class Maze:

    # actions
    RIGHT   = 0
    UP      = 1
    LEFT    = 2
    DOWN    = 3
    STAY    = 4

    # rewards
    R_STEP          = 0
    R_GOAL          = 50
    R_IMPOSSIBLE    = -100
    R_DIE           = -100

    def __init__(self, maze):
        '''

        :param maze:
        '''
        self.maze                       = maze
        self.actions                    = self.__actions()
        self.states, self.map           = self.__states()
        self.n_actions                  = len(self.actions)
        self.n_states                   = len(self.states)
        self.minotaur_mov               = self.__minotaur_movement()
        self.n_minotaur_mov             = len(self.minotaur_mov)
        self.transition_probabilities   = self.__transitions()
        self.rewards                    = self.__rewards()


    def __actions(self):
        actions = dict()
        actions[self.RIGHT] = (0, 1)
        actions[self.UP]    = (-1,0)
        actions[self.LEFT]  = (0,-1)
        actions[self.DOWN]  = (1, 0)
        actions[self.STAY]  = (0, 0)
        return actions

    def __minotaur_movement(self):
        minotaur = dict()
        minotaur[self.RIGHT] = (0, 1)
        minotaur[self.UP]    = (-1,0)
        minotaur[self.LEFT]  = (0,-1)
        minotaur[self.DOWN]  = (1, 0)
        return minotaur

    def __states(self):
        states = dict()
        map = dict()
        s = 0
        # agent position
        for row in range(self.maze.shape[0]):
            for col in range(self.maze.shape[1]):
                if self.maze[row,col] != 1:
                    # minotaur position
                    # with the assumption it can walk through wall
                    # but cannot not stay inside
                    for m_row in range(self.maze.shape[0]):
                        for m_col in range(self.maze.shape[1]):
                            if self.maze[row,col] != 1:
                                states[s] = (row,col,m_row,m_col)
                                map[(row,col,m_row,m_col)] = s
                                s += 1

        return states, map

    def __step_minotaur(self, pos, step):
        # step
        r = pos[0] + step[0]
        c = pos[1] + step[1]

        # hitting the edge:
        edge = (r == -1) or (r == self.maze.shape[0]) or \
               (c == -1) or (c == self.maze.shape[1])
        if edge:
            return False, [0,0]

        # going through wall
        elif self.maze[r,c] == 1:
            valid, pos = self.__step_minotaur([r,c],step)
            if not valid:
                return False, [0,0]
            else:
                return True, pos

        # feasible movement
        return True, [r,c]

    def __move(self, state, action, det_minotaur_movement=None):
        '''

        :param state:
        :param action:
        :param offline: switch to use this function
        :return:
        '''
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]

        # agent hits wall or edge
        if (row == -1) or (row == self.maze.shape[0]) or \
            (col == -1) or (col == self.maze.shape[1]) or \
            (self.maze[row,col] == 1):

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


    def __transitions(self):
        dim = (self.n_states, self.n_states, self.n_actions)
        transitions = np.zeros(dim)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = []
                for m in range(self.n_minotaur_mov):
                    # next_s = self.__move(s,a,m)
                    # if next_s not in next_states:
                    #     next_states.append(next_s)
                    next_states.append(self.__move(s,a,m))
                for next_s in next_states:
                    transitions[next_s, s, a] = 1/len(next_states)

        return transitions

    def __rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                for m_mov in range(self.n_minotaur_mov):
                    s_next = self.__move(s,a,m_mov)
                    # minotaur did not move
                    if self.states[s][2:4] == self.states[s_next][2:4]:
                        rewards[s,a] = self.R_IMPOSSIBLE
                    # agent hits the wall
                    elif self.states[s][0:2] == self.states[s_next][0:2] and a != self.STAY:
                        rewards[s,a] = self.R_IMPOSSIBLE
                    # minotaur catches the agent
                    elif self.states[s_next][0:2] == self.states[s_next][2:4]:
                        rewards[s,a] = self.R_DIE
                    # reward for reaching exit
                    elif self.states[s][0:2] == self.states[s_next][0:2] \
                            and self.maze[self.states[s_next][0:2]] == 2 \
                            and a == self.STAY:
                        rewards[s,a] = self.R_GOAL
                    else:
                        rewards[s,a] = self.R_STEP

        return rewards

    def simulate(self, start, policy):
        path = []
        win  = False
        # Initialize current state and time
        time = 0
        s = self.map[start]
        # Add the starting position in the maze to the path
        path.append(start)
        while True:
            # Move to next state given the policy and the current state
            if len(policy.shape) == 2:
                if time >= policy.shape[1]:
                    break
                next_s = self.__move(s, policy[s, time])
            else:
                next_s = self.__move(s, policy[s])

            next_pos = self.states[next_s]

            # Add the position in the maze corresponding to the next state
            # to the path
            path.append((next_pos))

            # minotaur catches the agent
            if next_pos[0:2] == next_pos[2:4]:
                break
            # winning
            elif self.maze[next_pos[0:2]] == 2:
                win = True
                break

            # Update time and state for next iteration
            time += 1
            s = next_s

        return path, win

    def get_grid_values_for_minpos(self, optimal_values, minotaur_pos, default_val):
        grid_val = np.zeros((self.maze.shape))

        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                # wall positions are not states
                if self.maze[r, c] != 1:
                    s = self.map[(r,c,minotaur_pos[0],minotaur_pos[1])]
                    grid_val[r,c] = optimal_values[s]
                else:
                    grid_val[r,c] = default_val

        return grid_val


    def animate_simulation(self, renderer, path, policy=None, V=None):
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
            renderer.update(agent_pos, minotaur_pos, policy_t_minpos, values_t_minpos, 0.3)

            # step
            t += 1


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    # Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming backwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1)
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)
    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def problem_1_b_dynprog(maze, renderer):
    T = 20
    start = (0, 0, 6, 5)

    V, policy = dynamic_programming(maze, T)

    path, win = maze.simulate(start, policy)

    maze.animate_simulation(renderer, path, policy, V)

    n_games = 10000
    n_win   = 0

    for game in range(n_games):
        path, win = maze.simulate(start, policy)
        if win:
            n_win += 1

    probability_of_winning = n_win/n_games

    print(probability_of_winning)

    # horizons = np.arange(20) + 1

def problem_1_b_valueiter(maze, renderer):
    gamma = 0.95
    epsilon = 0.0001
    start = (0, 0, 6, 5)

    V, policy = value_iteration(maze, gamma, epsilon)

    path, win = maze.simulate(start, policy)

    maze.animate_simulation(renderer, path, policy, V)

    n_games = 10000
    n_win = 0

    for game in range(n_games):
        path, win = maze.simulate(start, policy)
        if win:
            n_win +=1

    probability_of_winning = n_win/n_games
    print(probability_of_winning)

def problem_1_c():
    pass


if __name__ == '__main__':
    # False to plot, true to save images
    save_mode = False

    maze = Maze(DEF_MAZE)
    renderer = MazeRenderer(maze, save_mode)

    # running code for (b)
    # problem_1_b_dynprog(maze, renderer)

    # running code for (b) with value iteration
    problem_1_b_valueiter(maze, renderer)

    # running code for (c)
    problem_1_c()



