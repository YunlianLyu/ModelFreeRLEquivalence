import os
import random
import numpy as np
from datetime import datetime

from environments.riverswim import (
    RiverswimEnv,
    RiverswimEnvKappa1,
    RiverswimEnvKappa2,
    RiverswimEnvKappa3
)
from environments.gridworld import GridworldEnv, GridworldEnvKappa1
from agents.qlearning import qlearning, qlearning_es, qlearning_es_ucb, qlearning_ucb
from utils import save_obj, equivalenceClasses, profile_mapping, dict_max_v


def run_exp1(
        algo,
        env,
        env_name,
        save_info,
        n_steps=100000,
        gamma=0.85,
        epsilon=0.1,
        c=0.55,
        n_steps_store=10,
        n_runs=50,
        limit_q_value=False,
        c_lr=10.0
):
    """
    exp1: compute q-values for different types of q-learning algorithms
    on different environments.
    """

    saved_name = "./results/{}/{}".format(env_name, algo)
    if not os.path.exists(saved_name):
        os.makedirs(saved_name)

    if algo == "qlearning":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning(env, n_steps, gamma, epsilon, n_steps_store, c_lr, limit_q_value)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            q_max, t_q_max = dict_max_v(Q)
            print("max Q:", q_max)
            print("t max Q:", t_q_max)
            print(env_name)
            print(algo)
            print(i_seed)
            print()
    elif algo == "qlearning_ucb":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_ucb(env, n_steps, gamma, epsilon, c, n_steps_store)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            q_max, t_q_max = dict_max_v(Q)
            print("max Q:", q_max)
            print("t max Q:", t_q_max)
            print(env_name)
            print(algo)
            print(i_seed)
            print()
    elif algo == "qlearning_es":
        if env_name == "riverswim{}kappa1".format(env.nS):
            C_s_a, _ = equivalenceClasses(env, 0.1)
            sigma_s_a = profile_mapping(env, C_s_a)
            # C_s_a, _ = equivalenceClasses(RiverswimEnv(nS=env.nS), 0.0)
            # sigma_s_a = profile_mapping(RiverswimEnv(nS=env.nS), C_s_a)
        elif env_name == "riverswim{}kappa2".format(env.nS):
            C_s_a, _ = equivalenceClasses(env, 0.3)
            sigma_s_a = profile_mapping(env, C_s_a)
            # C_s_a, _ = equivalenceClasses(RiverswimEnv(nS=env.nS), 0.0)
            # sigma_s_a = profile_mapping(RiverswimEnv(nS=env.nS), C_s_a)
        elif env_name == "riverswim{}kappa3".format(env.nS):
            C_s_a, _ = equivalenceClasses(env, 0.3)
            sigma_s_a = profile_mapping(env, C_s_a)
            # C_s_a, _ = equivalenceClasses(RiverswimEnv(nS=env.nS), 0.0)
            # sigma_s_a = profile_mapping(RiverswimEnv(nS=env.nS), C_s_a)
        elif env_name == "gridworld2room_7_7_kappa1_5":
            C_s_a, _ = equivalenceClasses(env, 0.1)
            sigma_s_a = profile_mapping(env, C_s_a)
        else:
            C_s_a, _ = equivalenceClasses(env, 0.0)
            sigma_s_a = profile_mapping(env, C_s_a)
        print("equivalence structure:")
        for eqc in C_s_a:
            print(eqc)
        print("-" * 100)
        save_info += "classes:\n{}\n".format(np.array(C_s_a, dtype=object))
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_es(
                env, sigma_s_a, C_s_a, n_steps, gamma, epsilon, n_steps_store, c_lr, limit_q_value
            )
            # current problem: Q values are very high when nS is large for QL-ES in riverswim
            q_max, t_q_max = dict_max_v(Q)
            print("max Q:", q_max)
            print("t max Q:", t_q_max)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            print(algo)
            print(i_seed)
            print()
    elif algo == "qlearning_es_ucb":
        if env_name == "riverswim{}kappa1".format(env.nS):
            C_s_a, _ = equivalenceClasses(env, 0.1)
            sigma_s_a = profile_mapping(env, C_s_a)
            # C_s_a, _ = equivalenceClasses(RiverswimEnv(nS=env.nS), 0.0)
            # sigma_s_a = profile_mapping(RiverswimEnv(nS=env.nS), C_s_a)
        elif env_name == "riverswim{}kappa2".format(env.nS):
            C_s_a, _ = equivalenceClasses(env, 0.3)
            sigma_s_a = profile_mapping(env, C_s_a)
            # C_s_a, _ = equivalenceClasses(RiverswimEnv(nS=env.nS), 0.0)
            # sigma_s_a = profile_mapping(RiverswimEnv(nS=env.nS), C_s_a)
        elif env_name == "riverswim{}kappa3".format(env.nS):
            C_s_a, _ = equivalenceClasses(env, 0.3)
            sigma_s_a = profile_mapping(env, C_s_a)
            # C_s_a, _ = equivalenceClasses(RiverswimEnv(nS=env.nS), 0.0)
            # sigma_s_a = profile_mapping(RiverswimEnv(nS=env.nS), C_s_a)
        elif env_name == "gridworld2room_7_7_kappa1_5":
            C_s_a, _ = equivalenceClasses(env, 0.1)
            sigma_s_a = profile_mapping(env, C_s_a)
        else:
            C_s_a, _ = equivalenceClasses(env, 0.0)
            sigma_s_a = profile_mapping(env, C_s_a)
        print("equivalence structure:")
        for eqc in C_s_a:
            print(eqc)
        print("-" * 100)
        save_info += "classes:\n{}\n".format(np.array(C_s_a, dtype=object))
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_es_ucb(
                env, sigma_s_a, C_s_a, n_steps, gamma, epsilon, c, n_steps_store
            )
            q_max, t_q_max = dict_max_v(Q)
            print("max Q:", q_max)
            print("t max Q:", t_q_max)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            print(algo)
            print(i_seed)
            print()

    save_info += "end run time: {}".format(datetime.now())
    with open(os.path.join(saved_name, 'info.txt'), 'w') as file:
        file.write(save_info)


if __name__ == '__main__':
    # envs: riverswim, riverswimkappa1, riverswimkappa2, riverswimkappa3,
    #       gridworld, gridworldkappa1
    env_type = 'gridworld'

    if env_type == 'riverswim':
        nbstates = 6  # size of env
        env = RiverswimEnv(nS=nbstates)
        # env_name = env_type + str(nbstates)
        env_name = env_type + str(nbstates) + "_11"
    elif env_type == 'riverswimkappa1':
        nbstates = 20
        env = RiverswimEnvKappa1(nS=nbstates)
        env_name = "riverswim{}kappa1".format(nbstates)
    elif env_type == 'riverswimkappa2':
        nbstates = 20
        env = RiverswimEnvKappa2(nS=nbstates)
        env_name = "riverswim{}kappa2".format(nbstates)
    elif env_type == 'riverswimkappa3':
        nbstates = 20
        env = RiverswimEnvKappa3(nS=nbstates)
        env_name = "riverswim{}kappa3".format(nbstates)
    elif env_type == "gridworld":
        sizeX, sizeY = 11, 11
        # room name: 2room or 4room
        room_name = '4room'
        env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
        # env_name = env_type + room_name + "_" + str(sizeX) + "_" + str(sizeY)
        env_name = env_type + room_name + "_" + str(sizeX) + "_" + str(sizeY) + "_4"
    elif env_type == "gridworldkappa1":
        sizeX, sizeY = 7, 7
        # room name: 2room or 4room
        room_name = '2room'
        # bad resluts if use GridworldEnv for kappa test
        # env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
        env = GridworldEnvKappa1(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
        # env_name = "gridworld{}_{}_{}_{}".format(room_name, sizeX, sizeY, "kappa1")
        env_name = "gridworld{}_{}_{}_{}_{}".format(room_name, sizeX, sizeY, "kappa1", 5)

    # all algorithms
    # qlearning, qlearning_es, qlearning_ucb, qlearning_es_ucb
    algo = "qlearning_ucb"

    # parameters for each environment
    if env_name in ['riverswim6', 'riverswim6kappa1', 'riverswim6kappa2', 'riverswim6kappa3']:
        n_steps = 100000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55  # ucb constant
        n_steps_store = 50
    if env_name.split("_")[0] == "riverswim6":
        n_steps = 100000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 50
    elif env_name == 'riverswim8':
        n_steps = 120000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 60
    elif env_name == 'riverswim10':
        n_steps = 200000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 100
    elif env_name in ['riverswim20', 'riverswim20kappa1', 'riverswim20kappa2', 'riverswim20kappa3']:
        n_steps = 400000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 200
    elif env_name.split("_")[0] == 'riverswim20':
        n_steps = 400000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 200
    elif env_name == 'riverswim25':
        n_steps = 500000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 125
    elif env_name == 'riverswim40':
        n_steps = 800000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 400
    elif env_name.split("_")[0] == 'riverswim50':
        n_steps = 1000000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 500
    elif env_name in ['gridworld2room_7_7', 'gridworld2room_7_7_kappa1']:
        n_steps = 2000000
        gamma = 0.85
        epsilon = 0.2
        c = 0.06
        n_steps_store = 1000
    elif env_name[0:-2] == 'gridworld2room_7_7':
        n_steps = 1000000
        gamma = 0.85
        epsilon = 0.2
        c = 0.06
        n_steps_store = 500
    elif env_name[0:-2] == 'gridworld2room_7_7_kappa1':
        n_steps = 2000000
        gamma = 0.85
        epsilon = 0.2
        c = 0.06
        n_steps_store = 1000
    elif env_name == 'gridworld2room_9_9':
        n_steps = 2000000
        gamma = 0.85
        epsilon = 0.2
        c = 0.06
        n_steps_store = 1000
    elif env_name[0:-2] == 'gridworld2room_9_9_kappa1':
        n_steps = 1000000
        gamma = 0.85
        epsilon = 0.2
        c = 0.06
        n_steps_store = 500
    elif env_name == 'gridworld2room_11_11':
        n_steps = 2000000
        gamma = 0.85
        epsilon = 0.2
        c = 0.06
        n_steps_store = 1000
    elif env_name[0:-2] == 'gridworld2room_11_11':
        n_steps = 2000000
        gamma = 0.85
        epsilon = 0.5
        c = 0.06
        n_steps_store = 1000
    elif env_name == 'gridworld2room_13_13':
        n_steps = 2000000
        gamma = 0.85
        epsilon = 0.2
        c = 0.06
        n_steps_store = 1000
    elif (env_name == 'gridworld4room_7_7') or (env_name[0:-2] == 'gridworld4room_7_7'):
        n_steps = 1000000
        gamma = 0.85
        epsilon = 0.3
        c = 0.06
        n_steps_store = 500
    elif (env_name == 'gridworld4room_9_9') or (env_name[0:-2] == 'gridworld4room_9_9'):
        # n_steps = 4000000
        # n_steps = 2000000
        n_steps = 3000000
        gamma = 0.85
        epsilon = 0.3
        c = 0.06
        # n_steps_store = 2000
        # n_steps_store = 1000
        n_steps_store = 1500
    elif (env_name == 'gridworld4room_11_11') or (env_name[0:-2] == 'gridworld4room_11_11'):
        n_steps = 1000000
        gamma = 0.85
        epsilon = 0.1
        c = 0.06
        n_steps_store = 500
    elif (env_name == 'gridworld4room_13_13') or (env_name[0:-2] == 'gridworld4room_13_13'):
        n_steps = 2000000
        gamma = 0.85
        epsilon = 0.3
        c = 0.06
        n_steps_store = 1000

    n_runs = 100
    # current problem: super high value for qles in riverswim (ns>6)
    # to solve this: (1) one way is to limit q value to R_max/(1-\gamma) at each time step
    #                (2) second way is to set learning rate constant
    # can choose (1) or (2) to solve this problem
    # for ql and qles, limit q value or not
    limit_q_value = False  # set True if learning rate constant is high
    # for ql and qles, set learning rate coefficient
    c_lr = 0.1  # set to a small number if limit_q_value=False

    print("*" * 150)
    print("experiment infos:")
    print("algorithm:", algo, ", envname:", env_name, ", horizon:", n_steps, ", nruns:", n_runs, ", n_steps_store:",
          n_steps_store)
    print("gamma:", gamma, ", epsilon:", epsilon, ", c:", c)
    print("limit q value:", limit_q_value, ", learning rate:", c_lr)
    print("*" * 150)

    # save experimental settings
    save_exp_info = "experiment infos:\n"
    save_exp_info += "algorithm: {}\n".format(algo)
    save_exp_info += "envname: {}\n".format(env_name)
    save_exp_info += "horizon: {}\n".format(n_steps)
    save_exp_info += "nruns: {}\n".format(n_runs)
    save_exp_info += "nstepsstore: {}\n".format(n_steps_store)
    save_exp_info += "gamma: {}\n".format(gamma)
    save_exp_info += "epsilon: {}\n".format(epsilon)
    save_exp_info += "c: {}\n".format(c)
    save_exp_info += "limit q value: {}\n".format(limit_q_value)
    save_exp_info += "learning rate: {}\n".format(c_lr)
    save_exp_info += "start run time: {}\n".format(datetime.now())

    run_exp1(
        algo=algo,
        env=env,
        env_name=env_name,
        save_info=save_exp_info,
        n_steps=n_steps,
        gamma=gamma,
        epsilon=epsilon,
        c=c,
        n_steps_store=n_steps_store,
        n_runs=n_runs,
        limit_q_value=limit_q_value,
        c_lr=c_lr
    )
