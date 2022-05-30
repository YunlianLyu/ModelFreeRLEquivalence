from colorama import Fore, Style

from utils import q_value_iteration
from environments.riverswim import RiverswimEnv
from environments.gridworld import GridworldEnv, GridworldEnvKappa1

env = RiverswimEnv(nS=6)
print(Fore.CYAN + env.__class__.__name__ + str(env.nS))
q_optimal, policy_optimal = q_value_iteration(env=env, gamma=0.85)
print(Style.RESET_ALL + "optimal Q:")
print(q_optimal)
print("optimal policy:")
print(policy_optimal)
print("*" * 50)

sizeX, sizeY = 11, 11
room_name = '2room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
q_optimal, policy_optimal = q_value_iteration(env=env, gamma=0.98)
print(Style.RESET_ALL + "optimal Q:")
print(q_optimal)
print("optimal policy:")
print(policy_optimal)
print("*" * 50)

sizeX, sizeY = 7, 7
room_name = '2room'
env = GridworldEnvKappa1(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
q_optimal, policy_optimal = q_value_iteration(env=env, gamma=0.85)
print(Style.RESET_ALL + "optimal Q:")
print(q_optimal)
print("optimal policy:")
print(policy_optimal)
print("*" * 50)

sizeX, sizeY = 9, 9
room_name = '4room'
env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
print(Fore.CYAN + env.__class__.__name__ + room_name + "_" + str(sizeX) + "_" + str(sizeY))
q_optimal, policy_optimal = q_value_iteration(env=env, gamma=0.96)
print(Style.RESET_ALL + "optimal Q:")
print(q_optimal)
print("optimal policy:")
print(policy_optimal)
print("*" * 50)
