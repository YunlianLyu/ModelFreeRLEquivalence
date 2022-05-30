import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style('darkgrid')

from environments.riverswim import RiverswimEnv
from utils import q_value_iteration, plot_mean_CI
from plot import extract_data


def plot_results(env_name,
                 algo,
                 n_runs,
                 error,
                 plot_log_value,
                 q_optimal,
                 policy_optimal,
                 i_algo):
    # colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    #           "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    linestyle = "-"
    if algo == "qlearning_es":
        linestyle = "--"
    elif algo == "qlearning_es_ucb":
        linestyle = "-."

    plot_values = extract_data(n_runs, env_name, algo, error, plot_log_value, q_optimal, policy_optimal)

    if algo == "qlearning":
        legend = "QL"
        color = "tab:blue"
        marker = None
        marker_every = None
    if algo == "qlearning_1":
        legend = "QL1"
        color = "tab:blue"
        marker = None
        marker_every = None
    elif algo == "qlearning_es":
        legend = "QL-ES"
        color = "tab:green"
        marker = None
        marker_every = None
    elif algo == "qlearning_ucb":
        legend = "UCBQ"
        color = "tab:orange"
        marker = None
        marker_every = None
    elif algo == "qlearning_es_ucb":
        legend = "UCBQ-ES"
        color = "tab:red"
        marker = None
        marker_every = None
    elif algo == "qlearning_lambda":
        legend = "Q($\lambda$)"
        color = "tab:brown"
        marker = "o"
        marker_every = 10
    elif algo == "sarsa":
        legend = "Sarsa"
        color = "tab:purple"
        marker = "^"
        marker_every = 10
    elif algo == "sarsa_1":
        legend = "Sarsa"
        color = "tab:purple"
        marker = "^"
        marker_every = 10
    elif algo == "sarsa_lambda":
        legend = "Sarsa($\lambda$)"
        color = "tab:pink"
        marker = "s"
        marker_every = 10
    elif algo == "sarsa_lambda_1":
        legend = "Sarsa($\lambda$)"
        color = "tab:pink"
        marker = "s"
        marker_every = 10

    plot_mean_CI(plot_values,
                 legend,
                 T,
                 n_samping=12,
                 set_ci=set_ci,
                 set_semilog=set_semilog,
                 line_width=2,
                 line_style=linestyle,
                 color=color,
                 marker=marker,
                 marker_every=marker_every)


if __name__ == '__main__':
    env = RiverswimEnv(nS=6)

    env_type = "riverswim{}baselines".format(env.nS)

    gamma = 0.85
    q_optimal, policy_optimal = q_value_iteration(env, gamma=gamma)

    if env_type == "riverswim6baselines":
        T = 100000
    elif env_type == "riverswim8baselines":
        T = 120000

    algo_epsilon_decay = True
    plot_few = True

    # sarsa, sarsa_lambda, qlearning_lambda, qlearning, qlearning_es, qlearning_ucb, qlearning_ucb, qlearning_es_ucb
    if algo_epsilon_decay:
        algos = ["qlearning",
                 "qlearning_ucb",
                 # "qlearning_lambda",
                 "sarsa",
                 # "sarsa_lambda"
                 "qlearning_es",
                 "qlearning_es_ucb"
                 ]
    else:
        algos = ["qlearning",
                 "qlearning_ucb",
                 "qlearning_es",
                 "qlearning_es_ucb",
                 "qlearning_lambda",
                 "sarsa_1",
                 "sarsa_lambda_1"]

    # algos = ["qlearning",
    #          "qlearning_1",
    #          "sarsa",
    #          "sarsa_1",
    #          "sarsa_lambda",
    #          "sarsa_lambda_1"]

    n_runs = 100
    # error types: onenorm, infinorm, relainfinorm, gap
    plot_errors = ["onenorm",
                   "infinorm",
                   "relainfinorm",
                   "gap",
                   "nstates_optimal",
                   "nstates_suboptimal"]
    ylabels = ['$\ell_1$ absolute error',
               '$\ell_\infty$ absolute error',
               '$\ell_\infty$ relative error',
               'Suboptimality gap',
               'Optimal nstates',
               'Suboptimality nstates']
    i_error = 2
    error = plot_errors[i_error]
    print("env name:", env_type)
    print("error type:", error)

    plt.figure(figsize=(8, 6))
    plt.tight_layout()

    # plot settings
    set_ci = True  # first step, second step, third step
    plot_log_value = False  # second step
    set_yscale = False  # third step
    set_semilog = True  # fourth step

    if set_semilog:
        set_ci = False
    if set_yscale:
        plt.yscale('log')

    flag_ql = False
    flag_ucb = False
    flag_sarsa = False
    for algo in algos:
        if algo in ["qlearning", "qlearning_es", "qlearning_lambda"]:
            flag_ql = True
        elif algo in ["qlearning_ucb", "qlearning_es_ucb"]:
            flag_ucb = True
        elif algo in ['sarsa', "sarsa_1", "sarsa_lambda", "sarsa_lambda_1"]:
            flag_sarsa = True

    for i_algo, algo in enumerate(algos):
        print("algorithm:", algo)
        env_name = env_type
        plot_results(env_name=env_name,
                     algo=algo,
                     n_runs=n_runs,
                     error=error,
                     plot_log_value=plot_log_value,
                     q_optimal=q_optimal,
                     policy_optimal=policy_optimal,
                     i_algo=i_algo)

    if algo_epsilon_decay:
        if env_type == "riverswim6baselines":
            if plot_few:
                if plot_log_value:
                    if error == "infinorm":
                        ylims = (-4.5, 2)
                    elif error == "relainfinorm":
                        ylims = (-4.5, 1)
                elif set_yscale:
                    if error == "infinorm":
                        ylims = (10 ** (-1.8), 10 ** 1)
                    elif error == "relainfinorm":
                        ylims = (10 ** (-1.8), 10 ** 1)
                elif set_semilog:
                    if error == "infinorm":
                        ylims = (10 ** (-1.5), 10 ** 1)
                    elif error == "relainfinorm":
                        ylims = (10 ** (-1.5), 10 ** 1)
                else:
                    if error == "onenorm":
                        ylims = (0, 30)
                    elif error == "infinorm":
                        ylims = (0, 10)
                    elif error == "relainfinorm":
                        ylims = (0, 7)
                    elif error == "gap":
                        ylims = (-2.3, 0.1)
                    elif error == "nstates_optimal":
                        ylims = (0, 6)
                    elif error == "nstates_suboptimal":
                        ylims = (0, 6)
            else:
                if plot_log_value:
                    if error == "infinorm":
                        ylims = (-4.5, 2.5)
                    elif error == "relainfinorm":
                        ylims = (-4.5, 2)
                elif set_yscale:
                    if error == "infinorm":
                        ylims = (10 ** (-1.8), 10 ** 1.8)
                    elif error == "relainfinorm":
                        ylims = (10 ** (-1.8), 10 ** 1.7)
                elif set_semilog:
                    if error == "infinorm":
                        ylims = (10 ** (-1.5), 10 ** 1.6)
                    elif error == "relainfinorm":
                        ylims = (10 ** (-1.5), 10 ** 1.6)
                else:
                    if error == "onenorm":
                        ylims = (0, 50)
                    elif error == "infinorm":
                        ylims = (0, 30)
                    elif error == "relainfinorm":
                        ylims = (0, 18)
                    elif error == "gap":
                        ylims = (-30, 0.1)
                    elif error == "nstates_optimal":
                        ylims = (0, 6)
                    elif error == "nstates_suboptimal":
                        ylims = (0, 6)
        elif env_type == "riverswim8baselines":
            if plot_log_value:
                if error == "infinorm":
                    ylims = (-4, 2.5)
                elif error == "relainfinorm":
                    ylims = (-3.5, 2.5)
            elif set_yscale:
                if error == "infinorm":
                    ylims = (10 ** (-1.6), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.7)
            elif set_semilog:
                if error == "infinorm":
                    ylims = (10 ** (-1.5), 10 ** 1.6)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1), 10 ** 1.6)
            else:
                if error == "onenorm":
                    ylims = (0, 60)
                elif error == "infinorm":
                    ylims = (0, 12)
                elif error == "relainfinorm":
                    ylims = (0, 10)
                elif error == "gap":
                    ylims = (-0.6, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 8)
                elif error == "nstates_suboptimal":
                    ylims = (0, 8)
    else:
        if env_type == "riverswim6baselines":
            if plot_log_value:
                if error == "infinorm":
                    ylims = (-4.5, 2.5)
                elif error == "relainfinorm":
                    ylims = (-4.5, 2)
            elif set_yscale:
                if error == "infinorm":
                    ylims = (10 ** (-1.8), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.8), 10 ** 1.7)
            elif set_semilog:
                if error == "infinorm":
                    ylims = (10 ** (-1.5), 10 ** 1.6)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.5), 10 ** 1.6)
            else:
                if error == "onenorm":
                    ylims = (0, 60)
                elif error == "infinorm":
                    ylims = (0, 30)
                elif error == "relainfinorm":
                    ylims = (0, 18)
                elif error == "gap":
                    ylims = (-7, 0.1)
                elif error == "nstates_optimal":
                    ylims = (0, 6)
                elif error == "nstates_suboptimal":
                    ylims = (0, 6)
        elif env_type == "riverswim8baselines":
            if plot_log_value:
                if error == "infinorm":
                    ylims = (-4, 2.5)
                elif error == "relainfinorm":
                    ylims = (-3.5, 2.5)
            elif set_yscale:
                if error == "infinorm":
                    ylims = (10 ** (-1.6), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.7)
            elif set_semilog:
                if error == "infinorm":
                    ylims = (10 ** (-1.5), 10 ** 1.6)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1), 10 ** 1.6)
            else:
                if error == "onenorm":
                    ylims = (0, 60)
                elif error == "infinorm":
                    ylims = (0, 12)
                elif error == "relainfinorm":
                    ylims = (0, 10)
                elif error == "gap":
                    ylims = (-0.6, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 8)
                elif error == "nstates_suboptimal":
                    ylims = (0, 8)

    plt.ylim(ylims[0], ylims[1])
    plt.xlim(0, T)
    plt.ylabel(ylabels[i_error], fontsize=14)
    plt.xlabel('Time steps', fontsize=14)
    plt.title("RiverSwim", fontsize=15)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)

    fig_save_directory = "./results/plots/{}/".format(env_type)
    if not os.path.exists(fig_save_directory):
        os.makedirs(fig_save_directory)

    if error == "onenorm":
        note = "_sum"
    elif error == "infinorm":
        note = "_absolute"
    elif error == "relainfinorm":
        note = "_relative"
    elif error == "gap":
        note = "_gap"
    elif error == "nstates_optimal":
        note = "_nsoptimal"
    elif error == "nstates_suboptimal":
        note = "_nssuboptimal"

    draw_type = ""
    if algo_epsilon_decay:
        draw_type += "_decay"
    if flag_ql:
        draw_type += "_QL"
    if flag_ucb:
        draw_type += "_UCB"
    if flag_sarsa:
        draw_type += "_sarsa"
    if plot_log_value:
        draw_type += "_log"
    if set_semilog:
        draw_type += "_semilogy"
    if set_yscale:
        draw_type += "_yscale"
    fig_name = env_type + draw_type + note
    plt.savefig(fig_save_directory + fig_name + ".pdf")

    print("Finish plotting.")
