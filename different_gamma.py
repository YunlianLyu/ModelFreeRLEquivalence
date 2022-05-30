from datetime import datetime

from environments.riverswim import RiverswimEnv
from main import run_exp1

if __name__ == '__main__':
    # If Gamma is closer to zero, the agent will tend to consider only immediate rewards.
    # If Gamma is closer to one, the agent will consider future rewards with greater weight.

    env = RiverswimEnv(nS=6)

    gammas = [0.6, 0.7, 0.8, 0.85, 0.9]
    i_gamma = 2

    # qlearning, qlearning_es, qlearning_ucb, qlearning_es_ucb
    algo = "qlearning_es_ucb"

    n_steps = 100000
    gamma = gammas[i_gamma]
    epsilon = 0.1
    c = 0.55
    n_steps_store = 50

    n_runs = 100

    env_name = "riverswim{}gamma{}".format(env.nS, gammas[i_gamma])

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
