from colorama import Fore, Style

from environments.gridworld import GridworldEnv, GridworldEnvKappa1

sizeX, sizeY = 11, 11
room_name = '2room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "number of states:", len(env.P))
print("transition distribution:")
for k, v in env.P.items():
    print(k, v)
print("initial state distribution:", env.isd)
print("*" * 50)

sizeX, sizeY = 13, 13
room_name = '4room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "number of states:", len(env.P))
print("transition distribution:")
for k, v in env.P.items():
    print(k, v)
print("initial state distribution:", env.isd)
print("*" * 50)

sizeX, sizeY = 7, 7
room_name = '2room'
env = GridworldEnvKappa1(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
print(Style.RESET_ALL + "number of states:", len(env.P))
print("transition distribution:")
for k, v in env.P.items():
    print(k, v)
print("initial state distribution:", env.isd)
print("*" * 50)
