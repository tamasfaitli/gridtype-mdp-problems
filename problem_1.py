import numpy as np
from MazeRenderer import MazeRenderer


DEF_MAZE = np.array([[0,0,1,0,0,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,1,1],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,1,1,1,1,1,1,0],
                    [0,0,0,0,1,2,0,0]])

class Maze():

    # actions
    RIGHT   = 0
    UP      = 1
    LEFT    = 2
    DOWN    = 3
    STAY    = 4

    # rewards
    R_STEP          = -1
    R_GOAL          = 0
    R_IMPOSSIBLE    = -100

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
        actions[self.STAY]  = (0, 0)
        actions[self.RIGHT] = (0, 1)
        actions[self.UP]    = (-1,0)
        actions[self.LEFT]  = (0,-1)
        actions[self.DOWN]  = (1, 0)
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
                    # minotaur position (with the assumption it can walk inside walls)
                    for m_row in range(self.maze.shape[0]):
                        for m_col in range(self.maze.shape[1]):
                            states[s] = (row,col,m_row,m_col)
                            map[(row,col,m_row,m_col)] = s
                            s += 1

        return states, map

    def __move(self, state, action, m_mov):
        '''

        :param state:
        :param action:
        :return:
        '''
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][0]

        hitting_wall = (row == -1) or (row == self.maze.shape[0]) or \
                       (col == -1) or (col == self.maze.shape[1]) or \
                       (self.maze[row,col] == 1)

        m_row = self.states[state][2] + self.minotaur_mov[m_mov][0]
        m_col = self.states[state][3] + self.minotaur_mov[m_mov][1]

        m_hitting_edge = (m_row == -1) or (m_row == self.maze.shape[0]) or \
                         (m_col == -1) or (m_col == self.maze.shape[1])

        if hitting_wall:
            row = self.states[state][0]
            col = self.states[state][1]

        if m_hitting_edge:
            return state
        else:
            return self.map[(row,col,m_row,m_col)]


    def __transitions(self):
        dim = (self.n_states, self.n_states, self.n_actions)
        transitions = np.zeros(dim)

        for s in range(self.n_states):
            for a in range(self.n_actions):
                for m_mov in range(self.n_minotaur_mov):
                    s_next = self.__move(s,a,m_mov)
                    transitions[s_next, s, a] = 1

        return transitions

    def __rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                for m_mov in range(self.n_minotaur_mov):
                    s_next = self.__move(s,a,m_mov)
                    # hitting wall
                    if s == s_next and a != self.STAY:
                        rewards[s,a] = self.R_IMPOSSIBLE
                    # reward for reaching exit
                    elif s == s_next and self.maze[self.states[s_next][0:2]] == 2:
                        rewards[s,a] = self.R_GOAL
                    else:
                        rewards[s,a] = self.R_STEP

        return rewards



if __name__ == '__main__':
    maze = Maze(DEF_MAZE)