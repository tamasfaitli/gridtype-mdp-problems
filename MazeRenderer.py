#################################################
#                                               #
# EL2805 Reinforcement Learning                 #
# Computer Lab 1                                #
# Problem 1                                     #
#                                               #
# Author: Tamas Faitli (19960205-T410)          #
#                                               #
#################################################

import numpy as np
import matplotlib.pyplot as plt

# These assets are squared and it is expected to be squared (e.g. 40x40px)
AGENT_IMG       = 'res/agent_comp.npy'
MINOTAUR_IMG    = 'res/minotaur_comp.npy'
ARROW_IMG       = 'res/arrow_comp.npy'

class MazeRenderer():
    '''
    This class
    '''

    border_offset = 1
    exit_offset = 10

    td_offset = 6  # use even number
    htd_offset = int(td_offset / 2)
    r_offset = 2

    bar_width = 3

    def __init__(self, maze):
        '''
        Constructor of the renderer. Setting sizes, loading images.

        :param maze: numpy array defining maze structure
        '''
        self.skeleton       = maze.maze
        self.maze_shape     = maze.maze.shape

        self.agent_img      = np.load(AGENT_IMG)
        self.minotaur_img   = np.load(MINOTAUR_IMG)
        self.orig_arrow_img = np.load(ARROW_IMG) # right arrow

        # self.agent_pos      = agent_init
        # self.minotaur_pos   = minotaur_init

        assets = [self.agent_img, self.minotaur_img, self.orig_arrow_img]
        for a in assets:
            x, y = a.shape
            assert x==y, "Images should be squared shaped (e.g. 40x40px)!"

        assert self.agent_img.shape == self.minotaur_img.shape, \
            "Character images should be the same size!"

        self.grid_px    = self.agent_img.shape
        self.grid       = self.grid_px[0]
        self.action_px  = self.orig_arrow_img.shape

        self.__img_correction()

        # not the nicest way to fill this
        self.action_img = {maze.STAY    : np.zeros(self.action_px),
                           maze.RIGHT   : self.orig_arrow_img,
                           maze.UP      : np.rot90(self.orig_arrow_img),
                           maze.LEFT    : np.rot90(np.rot90(self.orig_arrow_img)),
                           maze.DOWN    : np.rot90(np.rot90(np.rot90(self.orig_arrow_img)))}

        self.plot_px    = np.multiply(self.grid_px, self.maze_shape)

        self.content    = np.ones((self.plot_px[0],self.plot_px[1], 4))

        self.canvas     = plt.figure(figsize=(6,5), frameon=False)

        self.bar_height = self.grid - self.td_offset

    def __img_correction(self):
        ''' Original image assets work with reversed gray colormap,
            this function can be used to adjust the images.

        :return: None
        '''
        char_reference  = np.ones(self.grid_px)
        arrow_reference = np.ones(self.action_px)

        self.agent_img      = char_reference - self.agent_img
        self.minotaur_img   = char_reference - self.minotaur_img
        self.orig_arrow_img = arrow_reference - self.orig_arrow_img

    def __reset_content(self):
        ''' Setting the plot to ones (clean white).

        :return:
        '''
        self.content[:,:,:] = 1.0

    def __fill_maze(self):
        ''' Fills the content with the maze

        :return:
        '''

        # drawing maze:
        for row in range(self.maze_shape[0]):
            for col in range(self.maze_shape[1]):
                # wall
                if self.skeleton[row,col] == 1:
                    self.content[row*self.grid:(row+1)*self.grid, \
                    col*self.grid:(col+1)*self.grid, 0:3] = 0.55

                # exit
                elif self.skeleton[row,col] == 2:
                    self.content[row*self.grid+self.exit_offset: \
                                 (row+1)*self.grid-self.exit_offset, \
                    col*self.grid+self.exit_offset: \
                    (col+1)*self.grid-self.exit_offset, 0:3:2] = 0.4
        # grid
        for row in range(self.maze_shape[0]):
            self.content[row*self.grid-1:row*self.grid+1,:,0:3] = 0.8
        for col in range(self.maze_shape[1]):
            self.content[:,col*self.grid-1:col*self.grid+1,0:3] = 0.8

    def __fill_character(self, img, pos):
        '''

        :param img:
        :param pos:
        :return:
        '''
        grid_posx = pos[0] * self.grid
        grid_posy = pos[1] * self.grid

        # for rgb in range(3):
        #     self.content[grid_posx:grid_posx+self.grid, \
        #     grid_posy:grid_posy+self.grid,rgb] = img

        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                for rgb in range(3):
                    self.content[grid_posx+x-1,grid_posy+y-1, rgb] = \
                        min(self.content[grid_posx+x-1,grid_posy+y-1, rgb],img[x,y])



    def __fill_policy(self, policy):
        '''

        :param policy: np.array with shape of the skeleton
                       mapping an action for each
        :return:
        '''
        assert policy.shape == self.maze_shape, "Policy cannot be plotted! Use 'None'"

        # drawing arrows
        for row in range(self.maze_shape[0]):
            for col in range(self.maze_shape[1]):
                act_img = self.action_img[policy[row,col]]

                # drawing arrows to bottom left
                for rgb in range(3):
                    self.content[-self.border_offset+(row+1)*self.grid-self.action_px[0]: \
                                 -self.border_offset+(row+1)*self.grid, \
                                self.border_offset+col*self.grid: \
                                self.border_offset+col*self.grid+self.action_px[1], \
                                rgb] = act_img


    def __fill_values(self, values):
        '''

        :param values: np.array with shape of the skeleton
                       describing the value function
        :return:
        '''
        assert values.shape == self.maze_shape, "Values cannot be plotted! Use 'None'"

        max_value = np.max(np.abs(values))

        # plotting values on the right side
        for row in range(self.maze_shape[0]):
            for col in range(self.maze_shape[1]):
                val = values[row,col]

                # positive values are green
                if val >= 0.0:
                    val_height = int((val/max_value)*self.bar_height)
                    # 0:3:2] = 0.0
                    self.content[-self.htd_offset+(row+1)*self.grid-val_height: \
                                 -self.htd_offset+(row+1)*self.grid, \
                                (col+1)*self.grid-(self.r_offset+self.bar_width): \
                                (col+1)*self.grid-self.r_offset, 0:3:2] = 0.0

                # else negative values are red
                else:
                    val_height = int((np.abs(val)/max_value)*self.bar_height)
                    # 1:3] = 0.0
                    self.content[-self.htd_offset + (row + 1) * self.grid - val_height: \
                                 -self.htd_offset + (row + 1) * self.grid, \
                    (col + 1) * self.grid - (self.r_offset + self.bar_width): \
                    (col + 1) * self.grid - self.r_offset, 1:3] = 0.0

    def __draw(self, pause):
        plt.imshow(self.content)
        plt.axis('off')
        plt.subplots_adjust(left=.0,right=1.0, top=1.0, bottom=0.0)

        plt.pause(pause)


    def update(self, agent_pos, minotaur_pos, policy=None, values=None, pause=0.1):
        # reset content
        self.__reset_content()

        # fill maze content
        self.__fill_maze()

        # fill agent content
        self.__fill_character(self.agent_img, agent_pos)

        # fill minotaur content
        self.__fill_character(self.minotaur_img, minotaur_pos)

        # fill policy
        if hasattr(policy, 'shape'):
            self.__fill_policy(policy)

        # fill values
        if hasattr(values, 'shape'):
            self.__fill_values(values)

        # draw image
        self.__draw(pause)

