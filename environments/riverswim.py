import numpy as np
from gym.envs.toy_text import discrete

LEFT = 0
RIGHT = 1


class RiverswimEnv(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self, nS):
        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnv, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            P[s][RIGHT] = [
                (0.05, max(0, s - 1), 0, 0),
                (0.55, s, 0, 0),
                (0.4, min(nS - 1, s + 1), 0, 0),
            ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class RiverswimEnvKappa1(discrete.DiscreteEnv):
    """ kappa-equivalent riverswim #1
    for (s_1,right) and (s_3,right), the transition is (0.05,0.45,0.5) instead of (0.05,0.55,0.4)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, nS=6):

        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnvKappa1, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            if s == 1 or s == 3:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.45, s, 0, 0),
                    (0.5, min(nS - 1, s + 1), 0, 0),
                ]
            else:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.55, s, 0, 0),
                    (0.4, min(nS - 1, s + 1), 0, 0),
                ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class RiverswimEnvKappa2(discrete.DiscreteEnv):
    """ kappa-equivalent riverswim #2
    for (s_1,right),the transition is (0.15,0.3,0.55) instead of (0.05,0.55,0.4)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, nS=6):

        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnvKappa2, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            if s == 1:
                P[s][RIGHT] = [
                    (0.15, max(0, s - 1), 0, 0),
                    (0.3, s, 0, 0),
                    (0.55, min(nS - 1, s + 1), 0, 0),
                ]
            else:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.55, s, 0, 0),
                    (0.4, min(nS - 1, s + 1), 0, 0),
                ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class RiverswimEnvKappa3(discrete.DiscreteEnv):
    """ kappa-equivalent riverswim #3
    for (s_1,right) and (s_3,right),the transition is (0.15,0.3,0.55) instead of (0.05,0.55,0.4)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, nS=6):

        # defining the number of actions
        nA = 2

        # defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)

        super(RiverswimEnvKappa3, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):

        # P[s][a] == [(probability,nextstate,reward,done),...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # reward transitions
        P[0][LEFT] = [(1.0, 0, 5 / 100, 0)]
        P[nS - 1][RIGHT] = [(0.6, nS - 1, 1, 0), (0.4, nS - 2, 1, 0)]

        # left transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1.0, max(0, s - 1), 0, 0)]

        # right transitions
        for s in range(1, nS - 1):
            if s == 1 or s == 3:
                P[s][RIGHT] = [
                    (0.15, max(0, s - 1), 0, 0),
                    (0.3, s, 0, 0),
                    (0.55, min(nS - 1, s + 1), 0, 0),
                ]
            else:
                P[s][RIGHT] = [
                    (0.05, max(0, s - 1), 0, 0),
                    (0.55, s, 0, 0),
                    (0.4, min(nS - 1, s + 1), 0, 0),
                ]
        P[0][RIGHT] = [(0.6, 0, 0, 0), (0.4, 1, 0, 0)]

        # initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        return P, isd

    def render(self, mode="human"):
        pass

    def close(self):
        pass
