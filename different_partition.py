import os
import random
import numpy as np
from datetime import datetime

from environments.riverswim import RiverswimEnv
from agents.qlearning import qlearning, qlearning_es, qlearning_ucb, qlearning_es_ucb
from utils import equivalenceClasses, profile_mapping, save_obj, dict_max_v


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
):
    """
    exp1: compute q-values for different types of q-learning algorithms
    on different environments.
    """

    if i_C == None:
        saved_name = "./results/{}/{}".format(env_name, algo)
    else:
        if algo == "qlearning_es":
            algo_name = "qles_partition" + C_names[i_C]
        elif algo == "qlearning_es_ucb":
            algo_name = "ucbqes_partition" + C_names[i_C]
        saved_name = "./results/{}/{}".format(env_name, algo_name)
    if not os.path.exists(saved_name):
        os.makedirs(saved_name)

    if algo == "qlearning":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning(env, n_steps, gamma, epsilon, n_steps_store)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print("max Q:", dict_max_v(Q))
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
            print("max Q:", dict_max_v(Q))
            print(env_name)
            print(algo)
            print(i_seed)
            print()
    elif algo == "qlearning_es":
        if i_C == None:
            C_s_a, _ = equivalenceClasses(env=env, eps=0.0)
        else:
            C_s_a = C[i_C]
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
                env, sigma_s_a, C_s_a, n_steps, gamma, epsilon, n_steps_store
            )
            # current problem: Q values are very high when nS is large for QL-ES in riverswim
            print("max Q:", dict_max_v(Q))
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            if i_C == None:
                print(algo)
            else:
                print(algo_name)
            print(i_seed)
            print()
    elif algo == "qlearning_es_ucb":
        if i_C == None:
            C_s_a, _ = equivalenceClasses(env=env, eps=0.0)
        else:
            C_s_a = C[i_C]
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
            print("max Q:", dict_max_v(Q))
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print(env_name)
            if i_C == None:
                print(algo)
            else:
                print(algo_name)
            print(i_seed)
            print()

    save_info += "end run time: {}".format(datetime.now())
    with open(os.path.join(saved_name, 'info.txt'), 'w') as file:
        file.write(save_info)


if __name__ == '__main__':
    # different ways to separate state-action pairs

    env = RiverswimEnv(nS=6)

    # C_s_a, _ = equivalenceClasses(env, eps=0.0)
    # for c in C_s_a:
    #     print(c)
    # print(type(C_s_a))
    """
    [(0, 1), (7, 1)]
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
    [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)]
    """

    if env.nS == 6:
        C1 = [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
              [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]]

        C2 = [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
              [(1, 1), (2, 1), (3, 1), (4, 1)],
              [(0, 1)],
              [(5, 1)]]

        C3 = [[(0, 0), (1, 0), (2, 0)],
              [(3, 0), (4, 0), (5, 0)],
              [(1, 1), (2, 1)],
              [(3, 1), (4, 1)],
              [(0, 1)],
              [(5, 1)]]

        C4 = [[(0, 0), (1, 0), (2, 0)],
              [(3, 0), (4, 0), (5, 0)],
              [(1, 1), (2, 1)],
              [(3, 1), (4, 1)],
              [(0, 1), (5, 1)]]
    elif env.nS == 8:
        C1 = [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
              [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]]

        C2 = [[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
              [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
              [(0, 1)],
              [(7, 1)]]

        C3 = [[(0, 0), (1, 0), (2, 0), (3, 0)],
              [(4, 0), (5, 0), (6, 0), (7, 0)],
              [(1, 1), (2, 1), (3, 1)],
              [(4, 1), (5, 1), (6, 1)],
              [(0, 1)],
              [(7, 1)]]

        C4 = [[(0, 0), (1, 0), (2, 0), (3, 0)],
              [(4, 0), (5, 0), (6, 0), (7, 0)],
              [(1, 1), (2, 1), (3, 1)],
              [(4, 1), (5, 1), (6, 1)],
              [(0, 1), (7, 1)]]

    C = [C1, C2, C3, C4]
    C_names = ["C1", "C2", "C3", "C4"]
    i_C = 3  # set None if run normal algos

    # qlearning, qlearning_es, qlearning_ucb, qlearning_es_ucb
    algo = "qlearning_es_ucb"

    if algo in ["qlearning", "qlearning_ucb"]:
        i_C = None

    if env.nS == 6:
        n_steps = 100000
        n_steps_store = 50
    elif env.nS == 8:
        n_steps = 120000
        n_steps_store = 60
    gamma = 0.85
    epsilon = 0.1
    c = 0.55

    n_runs = 100

    env_name = "riverswim{}partition".format(env.nS)

    print("*" * 100)
    print("experiment infos:")
    print("algorithm:", algo, ", envname:", env_name, ", horizon:", n_steps, ", nruns:", n_runs, ", n_steps_store:",
          n_steps_store)
    print("gamma:", gamma, ", epsilon:", epsilon, ", c:", c)
    print("*" * 100)

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
        n_runs=n_runs
    )
