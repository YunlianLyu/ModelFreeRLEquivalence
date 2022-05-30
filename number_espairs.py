from datetime import datetime
import numpy as np
import random
import os

from environments.riverswim import RiverswimEnv
from agents.qlearning import qlearning_es_few, qlearning_es_ucb_few
from utils import equivalenceClasses, profile_mapping, dict_max_v, save_obj
from main import run_exp1

if __name__ == '__main__':
    # choose few es-pairs to update at each time step
    env = RiverswimEnv(nS=6)

    env_name = "riverswim{}espairs_{}".format(env.nS, 2)

    if env.nS == 6:
        es_numbers = [1, 2, 3, 4]
    elif env.nS == 20:
        es_numbers = [3, 4, 5, 6, 7, 8, 10, 12, 16]
    i_number = 3
    n_es_pairs = es_numbers[i_number]

    # qlearning_es_few, qlearning_es_ucb_few
    algo = "qlearning_es_few"

    if env.nS == 6:
        n_steps = 100000
        n_steps_store = 50
    elif env.nS == 20:
        n_steps = 400000
        n_steps_store = 200
    gamma = 0.85
    epsilon = 0.1
    c = 0.55

    n_runs = 100
    limit_q_value = False
    c_lr = 4.0

    print("*" * 150)
    print("experiment infos:")
    print("algorithm:", algo, ", envname:", env_name, ", horizon:", n_steps, ", nruns:", n_runs, ", n_steps_store:",
          n_steps_store)
    print("gamma:", gamma, ", epsilon:", epsilon, ", c:", c, ", n_es_pairs:", n_es_pairs)
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
    save_exp_info += "limit q value:{}\n".format(limit_q_value)
    save_exp_info += "learning rate:{}\n".format(c_lr)
    save_exp_info += "nespairs: {}\n".format(es_numbers[i_number])
    save_exp_info += "start run time: {}\n".format(datetime.now())

    saved_name = "./results/{}/{}{}".format(env_name, algo, n_es_pairs)
    if not os.path.exists(saved_name):
        os.makedirs(saved_name)

    if algo == "qlearning_es_few":
        C_s_a, _ = equivalenceClasses(env, 0.0)
        sigma_s_a = profile_mapping(env, C_s_a)
        print("equivalence structure:")
        for eqc in C_s_a:
            print(eqc)
        print("-" * 100)
        save_exp_info += "classes:\n{}\n".format(np.array(C_s_a, dtype=object))
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_es_few(
                env, sigma_s_a, C_s_a, n_steps, gamma, epsilon, n_steps_store, n_es_pairs, c_lr, limit_q_value
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
        save_exp_info += "end run time: {}".format(datetime.now())
        with open(os.path.join(saved_name, 'info.txt'), 'w') as file:
            file.write(save_exp_info)
    elif algo == "qlearning_es_ucb_few":
        C_s_a, _ = equivalenceClasses(env, 0.0)
        sigma_s_a = profile_mapping(env, C_s_a)
        print("equivalence structure:")
        for eqc in C_s_a:
            print(eqc)
        print("-" * 100)
        save_exp_info += "classes:\n{}\n".format(np.array(C_s_a, dtype=object))
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_es_ucb_few(
                env, sigma_s_a, C_s_a, n_steps, gamma, epsilon, c, n_steps_store, n_es_pairs
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
        save_exp_info += "end run time: {}".format(datetime.now())
        with open(os.path.join(saved_name, 'info.txt'), 'w') as file:
            file.write(save_exp_info)
    else:
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
