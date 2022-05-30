from colorama import Fore, Style

from environments.riverswim import (
    RiverswimEnv,
    RiverswimEnvKappa1,
    RiverswimEnvKappa2,
    RiverswimEnvKappa3
)
from environments.gridworld import GridworldEnv, GridworldEnvKappa1
from utils import equivalenceClasses

nbstates = 20
env = RiverswimEnv(nS=nbstates)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.0)
print(Fore.CYAN + env.__class__.__name__ + str(nbstates))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("*" * 100)

nbstates = 20
env = RiverswimEnvKappa1(nS=nbstates)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.1)
print(Fore.CYAN + env.__class__.__name__ + str(nbstates))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("*" * 100)

env = RiverswimEnvKappa3(nS=nbstates)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.3)
print(Fore.CYAN + env.__class__.__name__ + str(nbstates))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("*" * 100)

sizeX, sizeY = 9, 9
room_name = '2room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.0)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("*" * 100)

env = GridworldEnvKappa1(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.0)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("*" * 100)

env = GridworldEnvKappa1(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
eqClasses, indexEqClass = equivalenceClasses(env=env, eps=0.1)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "equivalence classes:")
for c in eqClasses:
    print(c)
print("The number of classes:", len(eqClasses))
print("*" * 100)
