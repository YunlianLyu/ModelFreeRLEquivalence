import os
import numpy as np
import matplotlib.pyplot as plt


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
    maze[1][1] = 0.3
    maze[-2][-2] = 0.7
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
    maze[1][1] = 0.3
    maze[-2][-2] = 0.7
    return maze


if __name__ == '__main__':
    # maze_type: 2room, 4room
    maze_type = "2room"
    sizeX, sizeY = 13, 13

    if maze_type == "2room":
        maze = twoRoom(X=sizeX, Y=sizeY)
    elif maze_type == "4room":
        maze = fourRoom(X=sizeX, Y=sizeY)

    plt.figure()
    plt.imshow(maze, cmap='hot', interpolation='nearest')

    save_directory = "./results/envs/"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_name = "gridworld_{}_{}_{}".format(maze_type, sizeX, sizeY)
    plt.savefig(save_directory + save_name + '.pdf', bbox_inches='tight')

    print("Finish plotting " + save_name)
