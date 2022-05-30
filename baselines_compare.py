import os
from datetime import datetime
import numpy as np
import random

from environments.riverswim import RiverswimEnv
from agents.sarsa import sarsa, sarsa_es, sarsa_lambda
from agents.qlearning import qlearning, qlearning_es, qlearning_es_ucb, qlearning_ucb, qlearning_lambda
from utils import save_obj, dict_max_v, equivalenceClasses, profile_mapping
from main import run_exp1

if __name__ == '__main__':
    env_type = "riverswim"

    env = RiverswimEnv(nS=6)
    env_name = env_type + str(env.nS) + "baselines"

    use_eps_decay = True

    # sarsa, sarsa_es, sarsa_lambda, qlearning, qlearning_es, qlearning_ucb, qlearning_es_ucb, qlearning_labmda
    algo = "sarsa_es"

    if env_name == "riverswim6baselines":
        n_steps = 100000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 50
    elif env_name == "riverswim8baselines":
        n_steps = 120000
        gamma = 0.85
        epsilon = 0.1
        c = 0.55
        n_steps_store = 60

    n_runs = 100

    print("*" * 150)
    print("experiment infos:")
    print("algorithm:", algo, ", envname:", env_name, ", horizon:", n_steps, ", nruns:", n_runs, ", n_steps_store:",
          n_steps_store, "use epsilon decay:", use_eps_decay)
    print("gamma:", gamma, ", epsilon:", epsilon, ", c:", c)
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
    save_exp_info += "use epsilon decay: {}\n".format(use_eps_decay)
    save_exp_info += "start run time: {}\n".format(datetime.now())

    saved_name = "./results/{}/{}".format(env_name, algo)
    if not os.path.exists(saved_name):
        os.makedirs(saved_name)

    if algo == "sarsa":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = sarsa(env, n_steps, gamma, epsilon, n_steps_store, use_eps_decay)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print("max Q:", dict_max_v(Q))
            print(env_name)
            print(algo)
            print(i_seed)
            print()
        save_exp_info += "end run time: {}".format(datetime.now())
        with open(os.path.join(saved_name, 'info.txt'), 'w') as file:
            file.write(save_exp_info)
    elif algo == "sarsa_es":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            C_s_a, _ = equivalenceClasses(env, 0.0)
            sigma_s_a = profile_mapping(env, C_s_a)
            Q = sarsa_es(env, sigma_s_a, C_s_a, n_steps, gamma, epsilon, n_steps_store, use_eps_decay)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print("max Q:", dict_max_v(Q))
            print(env_name)
            print(algo)
            print(i_seed)
            print()
        save_exp_info += "end run time: {}".format(datetime.now())
        with open(os.path.join(saved_name, 'info.txt'), 'w') as file:
            file.write(save_exp_info)
    elif algo == "sarsa_lambda":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = sarsa_lambda(env, n_steps, gamma, epsilon, n_steps_store, use_eps_decay)
            # replace does not work well
            # Q = sarsa_lambda(env, n_steps, gamma, epsilon, n_steps_store, type='replace')
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print("max Q:", dict_max_v(Q))
            print(env_name)
            print(algo)
            print(i_seed)
            print()
        save_exp_info += "end run time: {}".format(datetime.now())
        with open(os.path.join(saved_name, 'info.txt'), 'w') as file:
            file.write(save_exp_info)
    elif algo == "qlearning_lambda":
        for i_seed in range(n_runs):
            np.random.seed(i_seed)
            random.seed(i_seed)
            env.seed(i_seed)
            Q = qlearning_lambda(env, n_steps, gamma, epsilon, n_steps_store)
            file_name = saved_name + "/q_{}".format(str(i_seed))
            save_obj(Q, file_name)
            print("max Q:", dict_max_v(Q))
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
