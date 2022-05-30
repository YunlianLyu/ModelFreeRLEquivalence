import numpy as np
from gym.envs.toy_text import discrete


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


class GridworldwithwallEnv(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, sizeX, sizeY, room_name='2room'):
        nS = sizeX * sizeY

        # defining the number of actions
        # 4 actions: 0=up, 1=down, 2=left, 3=right
        nA = 4

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(sizeX, sizeY, nS, nA, room_name)

        super(GridworldwithwallEnv, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, sizeX, sizeY, nS, nA, room_name):
        if room_name == '2room':
            maze = twoRoom(sizeX, sizeY)
        elif room_name == '4room':
            maze = fourRoom(sizeX, sizeY)

        massmap = [[0.7, 0.0, 0.06, 0.14, 0.1],  # up : up down left right stay
                   [0.0, 0.7, 0.06, 0.14, 0.1],  # down
                   [0.14, 0.06, 0.7, 0.0, 0.1],  # left
                   [0.06, 0.14, 0.0, 0.7, 0.1]]  # right

        goal_state_idx = sizeX * (sizeY - 1) - 2
        start_state_idx = sizeX + 2 - 1

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}
        # P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for s in range(nS):
            # send back to initial states when agent reaches goal state
            if s == goal_state_idx:
                for a in range(nA):
                    P[s][a].append((1.0, start_state_idx, 1, 0))
            else:
                # row and column index
                x, y = s // sizeY, s % sizeY
                # walls
                if maze[x, y] == 0:
                    for a in range(nA):
                        # P[s][a].append((1.0, s, 0, 0))
                        pass
                # true states except the goal state
                else:
                    us = [x - 1, y]
                    ds = [x + 1, y]
                    ls = [x, y - 1]
                    rs = [x, y + 1]
                    ss = [x, y]
                    if maze[us[0], us[1]] == 0: us = ss
                    if maze[ds[0], ds[1]] == 0: ds = ss
                    if maze[ls[0], ls[1]] == 0: ls = ss
                    if maze[rs[0], rs[1]] == 0: rs = ss
                    for a in range(nA):
                        p_s_a = np.zeros(nS)
                        next_states = set()
                        # add probs if block by walls
                        for k, e in enumerate([us, ds, ls, rs, ss]):
                            if massmap[a][k] != 0:
                                s_prime = e[0] * sizeX + e[1]
                                next_states.add(s_prime)
                                p_s_a[s_prime] += massmap[a][k]
                        # current problem: it changes float precision when taking values from p_s_a
                        for next_state in next_states:
                            prob = round(p_s_a[next_state], 2)
                            P[s][a].append((prob, next_state, 0, 0))

        # initial state distribution
        isd = np.zeros(nS)
        isd[start_state_idx] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def wallStates(self, sizeX, sizeY, maze):
        states = []
        walls = []
        for x in range(sizeX):
            for y in range(sizeY):
                sid = x * sizeX + y
                if maze[x, y] >= 1:
                    states.append(sid)
                else:
                    walls.append(sid)
        return states, walls

    def to_s(self, rowcol, sizeX):
        return rowcol[0] * sizeX + rowcol[1]

    def from_s(self, s, sizeX):
        return s // sizeX, s % sizeX
