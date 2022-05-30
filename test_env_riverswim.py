from colorama import Fore, Style

from environments.riverswim import (RiverswimEnv,
                                    RiverswimEnvKappa1,
                                    RiverswimEnvKappa2,
                                    RiverswimEnvKappa3)

nbstates = 20
env = RiverswimEnv(nS=nbstates)
print(Fore.CYAN + env.__class__.__name__ + str(env.nS))
print(Style.RESET_ALL + "transition distribution:")
for k, v in env.P.items():
    print(k, v)
print("initial state distribution:", env.isd)
print("*" * 50)

env = RiverswimEnvKappa1(nS=nbstates)
print(Fore.CYAN + env.__class__.__name__)
print(Style.RESET_ALL + "transition distribution:")
for k, v in env.P.items():
    print(k, v)
print("initial state distribution:", env.isd)
print("*" * 50)

env = RiverswimEnvKappa2(nS=nbstates)
print(Fore.CYAN + env.__class__.__name__)
print(Style.RESET_ALL + "transition distribution:")
for k, v in env.P.items():
    print(k, v)
print("initial state distribution:", env.isd)
print("*" * 50)

env = RiverswimEnvKappa3(nS=nbstates)
print(Fore.CYAN + env.__class__.__name__)
print(Style.RESET_ALL + "transition distribution:")
for k, v in env.P.items():
    print(k, v)
print("initial state distribution:", env.isd)
print("*" * 50)
