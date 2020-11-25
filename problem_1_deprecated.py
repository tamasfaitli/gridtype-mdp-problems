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

DEF_MAZE = np.array([[0,0,1,0,0,0,0,0],
                    [0,0,1,0,0,1,0,0],
                    [0,0,1,0,0,1,1,1],
                    [0,0,1,0,0,1,0,0],
                    [0,0,0,0,0,0,0,0],
                    [0,1,1,1,1,1,1,0],
                    [0,0,0,0,1,2,0,0]])

AS = 0
AR = 1
AU = 2
AL = 3
AD = 4
DEF_ACTION = {AS: 'stay', AR: 'right', AU: 'up', AL: 'left', AD: 'down'}

DEF_AGENT = 'agent_comp.npy'
DEF_MINOTAUR = 'minotaur_comp.npy'
DEF_ARROW = 'arrow_comp.npy'


# maze info, size and plots
class Maze():
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }


    def __init__(self, skeleton, exit_loc, scale):
        # the skeleton of the maze (structure, wall location etc...)
        self.skeleton = skeleton
        # number of row and number of colums within the maze
        self.Nr, self.Nc = np.shape(skeleton)
        # scaled row and columns
        self.Npx = scale*self.Nr
        self.Npy = scale*self.Nc
        self.scale = scale

        self.target = exit_loc

        # scaled version of the maze used for plots
        self.plot = np.zeros((self.Npx, self.Npy))

        # scaling the original skeleton with the size of the character images
        for r in range(self.Nr):
            for c in range(self.Nc):
                self.plot[r*self.scale:(r+1)*self.scale, c*self.scale:(c+1)*self.scale] = 0.6*self.skeleton[r,c]

        # drawing a grid
        for r in range(self.Nr):
            self.plot[r*self.scale-1:r*self.scale+1,:] = 0.4
        for c in range(self.Nc):
            self.plot[:,c*self.scale-1:c*self.scale+1] = 0.4


# character used for plots
class Character():
    def __init__(self, source, maze_size, init_pos):
        try:
            # pure content
            self.img = np.load(source)

        except Exception as e:
            print("Could not load character file: " + source + " !\n Exiting script...")
            exit(0)

        # position
        self.position = init_pos

        # number of row and number of colums within the maze
        self.scale = max(np.shape(self.img))

        # TODO handle non squared image (not important)

        # maze size
        self.maze_r = maze_size[0]
        self.maze_c = maze_size[1]

        # image size
        self.Nx, self.Ny = np.shape(self.img)

        # plot size
        self.Npx = self.scale*self.maze_r
        self.Npy = self.scale*self.maze_c

        # scaled version of the maze used for plots
        self.plot = np.zeros((self.Npx, self.Npy))

        self.update_position(self.position)

    def reset_plot(self):
        for x in range(self.Npx):
            for y in range(self.Npy):
                self.plot[x,y] = 0.0


    def update_position(self, pos):
        self.reset_plot()

        sposx = pos[0] * self.scale
        sposy = pos[1] * self.scale
        for x in range(self.Nx):
            for y in range(self.Ny):
                self.plot[sposx+x-1, sposy+y-1] = self.img[x,y]

class StateSpace():
    def __init__(self):
        self.state = np.array(4,1)

    def get_agent_position(self):
        return self.state[0:2]

    def get_minotaur_position(self):
        return self.state[2:4]


class ActionSpace():
    def __init__(self, maze, image):
        self.maze = maze.skeleton

        self.X,self.Y = np.shape(self.maze)

        self.actions = np.ones((self.X,self.Y,len(DEF_ACTION)))

        self.plot = np.zeros((maze.Npx,maze.Npy))
        self.scale = maze.scale
        self.Npx = maze.Npx
        self.Npy = maze.Npy

        ra = np.load(image)
        ua = np.rot90(ra)
        la = np.rot90(ua)
        da = np.rot90(la)
        s  = np.zeros((np.shape(ra)))

        # image size
        self.ix, self.iy = np.shape(ra)

        # images corresponding to actions to plot
        self.imgs = {AS: s, AR: ra, AU: ua, AL: la, AD: da}

        # create a bordered maze to easily define action space
        extended_maze = np.ones((self.X+2,self.Y+2))
        for x in range(self.X):
            for y in range(self.Y):
                extended_maze[x+1,y+1] = self.maze[x,y]

        # defining action space applying rules
        for nx in range(1, self.X+1):
            for ny in range(1, self.Y+1):
                # original maze indices
                x = nx-1
                y = ny-1

                # wall, no possible action
                if extended_maze[nx,ny] == 1:
                    self.actions[x,y,:] = 0
                    continue

                # wall below, no down movement
                if extended_maze[nx+1,ny] == 1:
                    self.actions[x,y,AD] = 0

                # wall to the right, no right movement
                if extended_maze[nx,ny+1] == 1:
                    self.actions[x,y,AR] = 0

                # wall to the left, no left movement
                if extended_maze[nx, ny-1] == 1:
                    self.actions[x,y,AL] = 0

                # wall above, no top movement
                if extended_maze[nx-1,ny] == 1:
                    self.actions[x,y,AU] = 0


        # preparing plot
        self.offset = self.scale - self.ix
        for x in range(self.X):
            for y in range(self.Y):
                actions_plot = np.zeros((np.shape(ra)))
                for a in range(len(DEF_ACTION)):
                    if self.actions[x,y,a] != 0:
                        actions_plot += self.imgs[a]

                # self.plot[x*self.scale:x*self.scale+self.ix, y*self.scale:y*self.scale+self.iy] = np.clip(actions_plot, 0.0,0.5)
                self.plot[x*self.scale+self.offset:x*self.scale+self.ix+self.offset, y*self.scale:y*self.scale+self.iy] = actions_plot


class Rewards():
    def __init__(self, maze):
        self.maze = maze.skeleton


class Policy():
    def __init__(self):
        pass


# handling image merging and updating plotting
class Renderer():
    def __init__(self, size):
        # content
        self.content = np.zeros((size))

        self.Nr = size[0]
        self.Nc = size[1]

        self.extent = 0, self.Nc, 0, self.Nr

        # facecolor = 'gray'
        self.figure = plt.figure(figsize=(6,5), frameon=False)
        # self.figure.tight_layout()

    def clip_pixel(self, value):
        if value > 1.0:
            value = 1.0
        return value

    def merge_plots(self, plots):
        for r in range(self.Nr):
            for c in range(self.Nc):
                pixel = 0.0
                for plot in plots:
                    pixel += plot[r,c]
                # self.content[r,c] = self.clip_pixel(pixel)
                self.content[r,c] = pixel

    def update_image(self, entities):
        plots = [x.plot for x in entities]
        self.merge_plots(plots)

        plt.imshow(self.content, cmap=plt.cm.gray_r, extent=self.extent, norm=plt.Normalize(vmax=1.0, vmin=0.0))
        plt.axis('off')
        plt.subplots_adjust(left=.0,right=1.0, top=1.0, bottom=0.0)

        pass










if __name__ == '__main__':
    agent = Character('res/' + DEF_AGENT, np.shape(DEF_MAZE), [0, 0])
    minotaur = Character('res/' + DEF_MINOTAUR, np.shape(DEF_MAZE), [6, 5])

    maze = Maze(DEF_MAZE, [6,5], agent.scale)

    renderer = Renderer((maze.Npx,maze.Npy))

    actionspace = ActionSpace(maze, 'res/'+DEF_ARROW)

    entities = [agent, minotaur, maze, actionspace]

    renderer.update_image(entities)
    for i in range(4):
        agent.update_position([i,i])
        renderer.update_image(entities)

        plt.pause(0.02)



    plt.show()