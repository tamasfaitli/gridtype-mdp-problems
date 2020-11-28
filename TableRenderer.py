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
ARROW_IMG       = 'res/arrow_comp.npy'

class TableRenderer:
    '''
    This class renders a grid style MDP problem with an agent and
    other character(s) on the table, with policy and value function.
    '''

    BORDER_OFFSET   = 1
    EXIT_OFFSET     = 10

    TD_OFFSET       = 6  # use even number
    HTD_OFFSET      = int(TD_OFFSET / 2)
    R_OFFSET        = 2

    BAR_WIDTH       = 3

    WALL            = 1
    GOAL            = 2
    OBJ             = 3

    def __init__(self, table, char_assets, save_mod=False, fig_size=(6,5)):
        ''' Constructor of the renderer. Setting sizes, loading images,
            init canvas.

        :param table:       numpy array defining table structure
        :param char_assets: dict: keys and image assets
        :param save_mod:    boolean whether to save or show images
        '''
        self.skeleton       = table.table
        self.table_shape    = table.table.shape

        # self.agent_img      = np.load(AGENT_IMG)
        # self.minotaur_img   = np.load(MINOTAUR_IMG)
        self.characters     = char_assets
        self.char_keys      = list(self.characters.keys())
        self.orig_arrow_img = np.load(ARROW_IMG) # right arrow

        assets = list(self.characters.values())
        assets.append(self.orig_arrow_img)
        for a in assets:
            x, y = a.shape
            assert x==y, "Images should be squared shaped (e.g. 40x40px)!"

        self.grid_px    = self.characters[self.char_keys[0]].shape
        self.grid       = self.grid_px[0]
        self.action_px  = self.orig_arrow_img.shape

        assert not any(img.shape != self.grid_px for img in list(self.characters.values())), \
            "Character images should be the same size!"

        self.__img_correction()

        # not the nicest way to fill this
        self.action_img = {table.STAY    : np.zeros(self.action_px),
                           table.RIGHT   : self.orig_arrow_img,
                           table.UP      : np.rot90(self.orig_arrow_img),
                           table.LEFT    : np.rot90(np.rot90(self.orig_arrow_img)),
                           table.DOWN    : np.rot90(np.rot90(np.rot90(self.orig_arrow_img)))}

        self.plot_px    = np.multiply(self.grid_px, self.table_shape)

        self.content    = np.ones((self.plot_px[0],self.plot_px[1], 4))

        self.canvas     = plt.figure(figsize=fig_size, frameon=False)

        self.bar_height = self.grid - self.TD_OFFSET

        self.save_mode  = save_mod
        self.saved_imgs = 0

    def __img_correction(self):
        ''' Original image assets work with reversed gray colormap,
            this function can be used to adjust the images.

        :return: None
        '''
        char_reference  = np.ones(self.grid_px)
        arrow_reference = np.ones(self.action_px)

        for char in self.char_keys:
            self.characters[char] = char_reference - self.characters[char]

        self.orig_arrow_img = arrow_reference - self.orig_arrow_img

    def __reset_content(self):
        ''' Setting the plot to ones (clean white).

        :return:
        '''
        self.content[:,:,:] = 1.0

    def __fill_table(self):
        ''' Fills the content with the maze

        :return:
        '''

        # drawing maze:
        for row in range(self.table_shape[0]):
            for col in range(self.table_shape[1]):
                # wall
                if self.skeleton[row,col] == self.WALL:
                    self.content[row*self.grid:(row+1)*self.grid, \
                    col*self.grid:(col+1)*self.grid, 0:3] = 0.55

                # exit
                elif self.skeleton[row,col] == self.GOAL:
                    self.content[row*self.grid+self.EXIT_OFFSET: \
                                 (row+1)*self.grid-self.EXIT_OFFSET, \
                    col*self.grid+self.EXIT_OFFSET: \
                    (col+1)*self.grid-self.EXIT_OFFSET, 0:3:2] = 0.4

                # other
                elif self.skeleton[row,col] == self.OBJ:
                    self.content[row * self.grid + self.EXIT_OFFSET: \
                                 (row + 1) * self.grid - self.EXIT_OFFSET, \
                    col * self.grid + self.EXIT_OFFSET: \
                    (col + 1) * self.grid - self.EXIT_OFFSET, 0:2:2] = 0.4
        # grid
        for row in range(self.table_shape[0]):
            self.content[row*self.grid-1:row*self.grid+1,:,0:3] = 0.8
        for col in range(self.table_shape[1]):
            self.content[:,col*self.grid-1:col*self.grid+1,0:3] = 0.8

    def __fill_character(self, img, pos):
        '''

        :param img:
        :param pos:
        :return:
        '''

        # scaled position in the content field
        grid_posx = pos[0] * self.grid
        grid_posy = pos[1] * self.grid

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
        assert policy.shape == self.table_shape, "Policy cannot be plotted! Use 'None'"

        # drawing arrows
        for row in range(self.table_shape[0]):
            for col in range(self.table_shape[1]):
                act_img = self.action_img[policy[row,col]]

                # drawing arrows to bottom left
                for rgb in range(3):
                    self.content[-self.BORDER_OFFSET + (row + 1) * self.grid - self.action_px[0]: \
                                 -self.BORDER_OFFSET + (row + 1) * self.grid, \
                    self.BORDER_OFFSET + col * self.grid: \
                                self.BORDER_OFFSET + col * self.grid + self.action_px[1], \
                                rgb] = act_img


    def __fill_values(self, values):
        '''

        :param values: np.array with shape of the skeleton
                       describing the value function
        :return:
        '''
        assert values.shape == self.table_shape, "Values cannot be plotted! Use 'None'"

        max_value = np.max(np.abs(values))

        # plotting values on the right side
        for row in range(self.table_shape[0]):
            for col in range(self.table_shape[1]):
                val = values[row,col]

                # positive values are green
                if val >= 0.0:
                    val_height = int((val/max_value)*self.bar_height)
                    # 0:3:2] = 0.0
                    self.content[-self.HTD_OFFSET + (row + 1) * self.grid - val_height: \
                                 -self.HTD_OFFSET + (row + 1) * self.grid, \
                                (col+1)*self.grid-(self.R_OFFSET + self.BAR_WIDTH): \
                                (col+1)*self.grid-self.R_OFFSET, 0:3:2] = 0.0

                # else negative values are red
                else:
                    val_height = int((np.abs(val)/max_value)*self.bar_height)
                    # 1:3] = 0.0
                    self.content[-self.HTD_OFFSET + (row + 1) * self.grid - val_height: \
                                 -self.HTD_OFFSET + (row + 1) * self.grid, \
                    (col + 1) * self.grid - (self.R_OFFSET + self.BAR_WIDTH): \
                    (col + 1) * self.grid - self.R_OFFSET, 1:3] = 0.0

    def __draw(self, pause):
        if self.save_mode:
            plt.imsave('res/fig' + str(self.saved_imgs) + '.png', self.content)
            self.saved_imgs += 1
        else:
            plt.imshow(self.content)
            plt.axis('off')
            plt.subplots_adjust(left=.0,right=1.0, top=1.0, bottom=0.0)
            plt.pause(pause)


    def update(self, state, policy=None, values=None, pause=0.1):
        ''' Update the current content and plot it or save it.

        :param state:   position for each character (char1_x, char1_y, char2_x, char2_y, ...)
        :param policy:  action index for each position on the grid
        :param values:  value for each position on the grid
        :param pause:   time till each frame being shown (in seconds)
        :return:
        '''
        # reset content
        self.__reset_content()

        # fill table content
        self.__fill_table()

        # fill character contents
        char = 0
        for row,col in zip(state[0::2], state[1::2]):
            self.__fill_character(self.characters[self.char_keys[char]], (row,col))
            char += 1

        # fill policy
        if hasattr(policy, 'shape'):
            self.__fill_policy(policy)

        # fill values
        if hasattr(values, 'shape'):
            self.__fill_values(values)

        # draw image
        self.__draw(pause)

    def set_save_mod(self, mode):
        self.save_mode = mode

    def reset_image_counter(self):
        self.saved_imgs = 0

