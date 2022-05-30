from colorama import Fore, Style

from environments.riverswim import (RiverswimEnv,
                                    RiverswimEnvKappa1,
                                    RiverswimEnvKappa2,
                                    RiverswimEnvKappa3)
from environments.gridworld import GridworldEnv, GridworldEnvKappa1
from utils import equivalenceClasses, profile_mapping

env = RiverswimEnv(nS=6)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.0)
print(Fore.CYAN + env.__class__.__name__ + str(env.nS))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("classes index:")
print(indexEqClass)
print("class probabilities:")
print("-" * 50)
for i, c in enumerate(eqClasses):
    print("class:", i)
    for s_a_pair in c:
        s, a = s_a_pair
        tran = env.P[s][a]
        print((s, a), tran)
print("-" * 50)
sigma = profile_mapping(env=env, C=eqClasses)
print("profile mapping (sigma):")
for k, v in sigma.items():
    print(k, v)
print("*" * 50)

env = RiverswimEnvKappa1(nS=20)
# eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.1)
eqClasses, indexEqClass = equivalenceClasses(env=RiverswimEnv(env.nS), eps=0.0)
print(Fore.CYAN + env.__class__.__name__ + str(env.nS))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("classes index:")
print(indexEqClass)
print("class probabilities:")
print("-" * 50)
for i, c in enumerate(eqClasses):
    print("class:", i)
    for s_a_pair in c:
        s, a = s_a_pair
        tran = env.P[s][a]
        print((s, a), tran)
print("-" * 50)
# sigma = profile_mapping(env=env, C=eqClasses)
sigma = profile_mapping(env=RiverswimEnv(nS=env.nS), C=eqClasses)
print("profile mapping (sigma):")
for k, v in sigma.items():
    print(k, v)
print("*" * 50)

env = RiverswimEnvKappa2()
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.3)
print(Fore.CYAN + env.__class__.__name__ + str(env.nS))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("classes index:")
print(indexEqClass)
print("class probabilities:")
print("-" * 50)
for i, c in enumerate(eqClasses):
    print("class:", i)
    for s_a_pair in c:
        s, a = s_a_pair
        tran = env.P[s][a]
        print((s, a), tran)
print("-" * 50)
sigma = profile_mapping(env=env, C=eqClasses)
print("profile mapping (sigma):")
for k, v in sigma.items():
    print(k, v)
print("*" * 50)

env = RiverswimEnvKappa3()
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.3)
print(Fore.CYAN + env.__class__.__name__ + str(env.nS))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("classes index:")
print(indexEqClass)
print("class probabilities:")
print("-" * 50)
for i, c in enumerate(eqClasses):
    print("class:", i)
    for s_a_pair in c:
        s, a = s_a_pair
        tran = env.P[s][a]
        print((s, a), tran)
print("-" * 50)
sigma = profile_mapping(env=env, C=eqClasses)
print("profile mapping (sigma):")
for k, v in sigma.items():
    print(k, v)
print("*" * 50)

sizeX, sizeY = 7, 7
room_name = '2room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.0)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("classes index:")
print(indexEqClass)
print("class probabilities:")
print("-" * 50)
for i, c in enumerate(eqClasses):
    print("class:", i)
    for s_a_pair in c:
        s, a = s_a_pair
        tran = env.P[s][a]
        print((s, a), tran)
print("-" * 50)
sigma = profile_mapping(env=env, C=eqClasses)
print("profile mapping (sigma):")
for k, v in sigma.items():
    print(k, v)
print("*" * 50)

sizeX, sizeY = 9, 9
room_name = '4room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.0)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("classes index:")
print(indexEqClass)
print("class probabilities:")
print("-" * 50)
for i, c in enumerate(eqClasses):
    print("class:", i)
    for s_a_pair in c:
        s, a = s_a_pair
        tran = env.P[s][a]
        print((s, a), tran)
print("-" * 50)
sigma = profile_mapping(env=env, C=eqClasses)
print("profile mapping (sigma):")
for k, v in sigma.items():
    print(k, v)
print("*" * 50)

sizeX, sizeY = 7, 7
room_name = '2room'
env = GridworldEnvKappa1(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.1)
print(Fore.CYAN + env.__class__.__name__ + "_" + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("classes index:")
print(indexEqClass)
print("class probabilities:")
print("-" * 50)
for i, c in enumerate(eqClasses):
    print("class:", i)
    for s_a_pair in c:
        s, a = s_a_pair
        tran = env.P[s][a]
        print((s, a), tran)
print("-" * 50)
sigma = profile_mapping(env=env, C=eqClasses)
print("profile mapping (sigma):")
for k, v in sigma.items():
    print(k, v)
print("*" * 50)
