import os
import matplotlib.pyplot as plt
import numpy as np

from environments.gridworldwithwall import GridworldwithwallEnv
from utils import equivalenceClasses


def twoRoom(X, Y):
    X2 = (int)(X / 2)
    maze = np.ones((X, Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y - 1] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X - 1][y] = 0.
        maze[X2][y] = 0.
    maze[X2][(int)(Y / 2)] = 1.
    return maze


def fourRoom(X, Y):
    Y2 = (int)(Y / 2)
    X2 = (int)(X / 2)
    maze = np.ones((X, Y))
    for x in range(X):
        maze[x][0] = 0.
        maze[x][Y - 1] = 0.
        maze[x][Y2] = 0.
    for y in range(Y):
        maze[0][y] = 0.
        maze[X - 1][y] = 0.
        maze[X2][y] = 0.
        maze[X2][(int)(Y2 / 2)] = 1.
        maze[X2][(int)(3 * Y2 / 2)] = 1.
        maze[(int)(X2 / 2)][Y2] = 1.
        maze[(int)(3 * X2 / 2)][Y2] = 1.
    return maze


def plotGridWorldEquivClasses(env, eqclasses, sizeX, sizeY, maze, folder="", numFigure=1):
    nbFigure = plt.gcf().number + 1
    plt.figure(nbFigure)
    plt.tight_layout()
    actions = ['Up', 'Down', 'Left', 'Right']  # 0=up, 1=down, 2=left, 3=right
    equiv0 = np.zeros((sizeX, sizeY))
    equiv1 = np.zeros((sizeX, sizeY))
    equiv2 = np.zeros((sizeX, sizeY))
    equiv3 = np.zeros((sizeX, sizeY))
    numq = 0
    eqClasses = sorted(eqclasses, key=lambda x: len(x))
    for eq in eqClasses:
        numq += 1
        for e in eq:
            x, y = env.from_s(e[0], sizeX)
            if (maze[x][y] > 0):
                if (e[1] == 0):
                    equiv0[x][y] = numq
                if (e[1] == 1):
                    equiv1[x][y] = numq
                if (e[1] == 2):
                    equiv2[x][y] = numq
                if (e[1] == 3):
                    equiv3[x][y] = numq
    f, axarr = plt.subplots(2, 2, gridspec_kw={'wspace': -0.4, 'hspace': 0.2})
    axarr[0, 0].imshow(equiv0, cmap='hot', interpolation='nearest', vmin=0, vmax=numq)
    axarr[0, 0].set_title(actions[0])
    axarr[0, 1].imshow(equiv1, cmap='hot', interpolation='nearest', vmin=0, vmax=numq)
    axarr[0, 1].set_title(actions[1])
    axarr[1, 0].imshow(equiv2, cmap='hot', interpolation='nearest', vmin=0, vmax=numq)
    axarr[1, 0].set_title(actions[2])
    axarr[1, 1].imshow(equiv3, cmap='hot', interpolation='nearest', vmin=0, vmax=numq)
    axarr[1, 1].set_title(actions[3])
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    # one liner to remove *all axes in all subplots*
    # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(folder, bbox_inches='tight')


if __name__ == '__main__':
    # maze_type: 2room, 4room
    maze_type = "4room"

    sizeX, sizeY = 9, 9

    if maze_type == "2room":
        maze = twoRoom(X=sizeX, Y=sizeY)
    elif maze_type == "4room":
        maze = fourRoom(X=sizeX, Y=sizeY)
    # print(maze)

    env = GridworldwithwallEnv(sizeX=sizeX, sizeY=sizeY, room_name=maze_type)

    C, _ = equivalenceClasses(env=env, eps=0.0)
    plot_es = True
    if plot_es:
        print("equivalence structure:")
        for c in C:
            print(c)
        print("The number of classes:", len(C))

    plot_probs = True
    if plot_probs:
        for c in C:
            print("-" * 100)
            for s_a_pair in c:
                s, a = s_a_pair
                print((s, a), env.P[s][a])
        print("-" * 100)

    save_directory = "./results/env_es/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_name = "classes_gridworld_{}_{}_{}.pdf".format(maze_type, sizeX, sizeY)
    plotGridWorldEquivClasses(env, C, sizeX, sizeY, maze, folder=(save_directory + save_name))

    print("Finish plotting ", save_name)
