import numpy as np
from datetime import datetime
from collections import defaultdict

from environments.riverswim import RiverswimEnv
from environments.gridworld import GridworldEnv
from utils import equivalenceClasses, profile_mapping

a = np.array([
    [1, 2, 3, 4, 5, 4, 6, 8, 2, 4, 2, 5, 3, 4, 6],
    [3, 4, 6, 1, 4, 4, 2, 6, 4, 9, 1, 2, 3, 4, 1],
    [2, 4, 5, 7, 2, 4, 2, 7, 4, 9, 2, 4, 2, 1, 5],
    [1, 3, 7, 2, 4, 1, 4, 2, 6, 3, 2, 3, 5, 2, 4],
    [3, 9, 8, 1, 2, 2, 4, 6, 7, 1, 2, 5, 2, 6, 2],
])
a_mean = np.mean(a, axis=0)
print(a_mean)
a_sde = 1.95 * np.std(a, axis=0) / np.sqrt(a.shape[0])
print(a_sde)
n_sampling = 2
T = a.shape[1]
print(T)
a_mean_sampling = a_mean[0:T:n_sampling]
print(a_mean_sampling)
a_sde_sampling = a_sde[0:T:n_sampling]
print(a_sde_sampling)
print("*" * 100)

x = np.array(range(0, T))[0:T:2]
print(x)
print("*" * 100)

time = datetime.now()
print(time)
info = ""
info += "start time: {}\n".format(datetime.now())
print(info)
c = np.array([
    [1, 2, 3],
    [5, 4, 7]
])
info += "classes:\n{}".format(c)
print(info)
print("*" * 100)

env = RiverswimEnv(nS=6)
C, _ = equivalenceClasses(env=env, eps=0.0)
C = np.array(C, dtype=object)
print(C)
info = ""
info += "classes:\n{}".format(C)
print(info)
print("*" * 100)

a = np.array([2, 23, 15, 7, 9, 11, 17, 19, 5, 3])
a[a > 10] = 0
print(a)
print(1e-6)
a = np.array([2, 23, 15, 7, -1, 11, 17, -5, 5, 3])
a[a <= 0] = 0
print(a)
print("*" * 100)

x = np.array([0.2, 6.4, 3.0, 1.6])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)
print(inds)
for n in range(x.size):
    print(bins[inds[n] - 1], "<=", x[n], "<", bins[inds[n]])
print("*" * 100)

env = RiverswimEnv(nS=6)
Q = {}
E = {}
for s in range(env.nS):
    Q[s] = [0] * env.nA
    E[s] = [0] * env.nA
# for k, v in Q.items():
#     print(k, v)
# eligibility = defaultdict(lambda: defaultdict(lambda: 0))
# for k, v in eligibility.items():
#     print(k, v)
E = defaultdict(lambda: np.zeros(env.nA))
E[0][0] += 1
for k, v in E.items():
    print(k, v)
print("*" * 50)

env = RiverswimEnv(nS=6)
C_s_a, _ = equivalenceClasses(env=env, eps=0.0)
sigma_s_a = profile_mapping(env=env, C=C_s_a)
for c in C_s_a:
    print(c)
print("all pairs:", sigma_s_a[1, 1, 2])
# index = range(len(sigma_s_a[1, 1, 2]))
# print(index)
# n_es_index = sorted(np.random.choice(index, 2, replace=False))
# print(n_es_index)
# espairs = []
# for i_index in n_es_index:
#     espairs.append(sigma_s_a[1, 1, 2][i_index])
# print(espairs)
# randomly choose n pairs to update each time
n_es_pairs = 2
all_pairs = sigma_s_a[1, 1, 2]
n_c = len(all_pairs)
if n_c <= n_es_pairs:
    es_pairs = all_pairs
else:
    index_all_pairs = range(n_c)
    index_es = sorted(np.random.choice(index_all_pairs, n_es_pairs, replace=False))
    print(index_es)
    es_pairs = []
    for i_index in index_es:
        es_pairs.append(all_pairs[i_index])
print("es pairs:", es_pairs)
n_c = len({})
print(n_c)
print("*" * 50)

a = np.array([[5],
              [9],
              [7],
              [6]])
b = np.array([[1, 6, 2],
              [15, 10, 8],
              [2, 5, 1],
              [3, 4, 1]])
print(a - b)
print(np.min(a - b, axis=1))
print(np.count_nonzero(np.min((a - b), axis=1) >= 0))
b_subopt = np.max(b, axis=1).reshape(-1, 1)
print(b_subopt)
print(a - b_subopt)
print(np.min(a - b_subopt))
print("*" * 50)

a = np.array([[12, 5, 7],
              [3, 6, 11],
              [8, 17, 9],
              [8, 10, 2]])
b = np.array([[1, 6, 2],
              [15, 10, 8],
              [2, 5, 1],
              [3, 4, 1]])
print(a - b)
print(np.abs(a - b))
print(np.max(np.abs(a - b)))
print((a - b) / a)
print(np.abs((a - b) / a))
print(np.max(np.abs((a - b) / a)))
print("*" * 50)

env = GridworldEnv(7, 7, "4room")
C, _ = equivalenceClasses(env, eps=0.0)
for c in C:
    print(c)
print("-" * 50)
C, _ = equivalenceClasses(env, eps=0.2)
for c in C:
    print(c)
print("*" * 50)

sizeX, sizeY = 7, 7
room_name = '2room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
env_name = "gridworld{}_{}_{}_{}".format(room_name, sizeX, sizeY, "kappa1")
print(env_name)
print("*" * 50)

a = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [2, 4, 5, 6, 8, 7, 9]
])
m, n = np.shape(a)
print(m, n)
a = np.array([a[i][0:4] for i in range(0, m)])
print(a)
print(np.linspace(0, 40))
print(np.linspace(0, 40, num=9))
print("*" * 100)

env = "gridworld2room_7_7_3"
print(env[0:-2])
x = [1, 2, 3, 4, 5, 6, 7, 8]
print(x[0:2])
