import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from environments.riverswim import RiverswimEnv
from utils import q_value_iteration, load_obj, plot_mean_CI
from plot import extract_data

sns.set_style('darkgrid')


def plot_results(env_name,
                 algo,
                 gamma,
                 n_runs,
                 error,
                 plot_log_value,
                 q_optimal,
                 policy_optimal):
    # all colors: tab:blue, tab:orange, tab:green, tab:red, tab:purple
    #             tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan

    # all line styles: -, --, -., :

    # makers: ".", "*", "x", "1", "|", "v", "d"

    if flag_ql and flag_ucb:
        if algo == "qlearning":
            color = "tab:blue"
            linestyle = '-'
        elif algo == "qlearning_es":
            linestyle = '--'
            color = "tab:green"
        elif algo == "qlearning_ucb":
            linestyle = '-'
            color = "tab:orange"
        elif algo == "qlearning_es_ucb":
            linestyle = '-.'
            color = "tab:red"

        if gamma == 0.6:
            # linestyle = ':'
            marker = "X"
        elif gamma == 0.7:
            # linestyle = '-.'
            marker = "v"
        elif gamma == 0.8:
            # linestyle = '-'
            marker = "."
        elif gamma == 0.85:
            # linestyle = '--'
            marker = None
        elif gamma == 0.9:
            # linestyle = '-'
            marker = "*"
    elif flag_ql:
        if gamma == 0.6:
            color = "tab:blue"
        elif gamma == 0.7:
            color = "tab:pink"
        elif gamma == 0.8:
            color = "tab:blue"
        elif gamma == 0.85:
            color = "tab:orange"
        elif gamma == 0.9:
            color = "tab:green"

        if algo == "qlearning":
            linestyle = '-'
            marker = "."
        elif algo == "qlearning_es":
            linestyle = '--'
            marker = "*"
        elif algo == "qlearning_ucb":
            linestyle = '-.'
            marker = "|"
        elif algo == "qlearning_es_ucb":
            linestyle = ':'
            marker = "v"

    plot_values = extract_data(n_runs, env_name, algo, error, plot_log_value, q_optimal, policy_optimal)

    if algo == "qlearning":
        legend = "QL,$\gamma={}$".format(gamma)
    elif algo == "qlearning_es":
        legend = "QL-ES,$\gamma={}$".format(gamma)
    elif algo == "qlearning_ucb":
        legend = "UCBQ,$\gamma={}$".format(gamma)
    elif algo == "qlearning_es_ucb":
        legend = "UCBQ-ES,$\gamma={}$".format(gamma)

    plot_mean_CI(plot_values,
                 legend,
                 T,
                 n_samping=14,
                 set_ci=set_ci,
                 set_semilog=set_semilog,
                 line_width=2,
                 line_style=linestyle,
                 color=color,
                 marker=marker,
                 marker_every=12)


if __name__ == '__main__':
    env = RiverswimEnv(nS=6)

    env_type = "riverswim{}gamma".format(env.nS)

    T = 100000

    # gammas = [0.6, 0.7, 0.8, 0.85, 0.9]
    gammas = [0.9, 0.85, 0.8]

    algos = ["qlearning", "qlearning_ucb", "qlearning_es", "qlearning_es_ucb"]
    # algos = ["qlearning", "qlearning_es"]

    n_runs = 100
    # error types: onenorm, infinorm, relainfinorm, gap
    plot_errors = ["onenorm", "infinorm", "relainfinorm", "gap", "nstates_optimal"]
    ylabels = ['$\ell_1$ absolute error',
               '$\ell_\infty$ absolute error',
               '$\ell_\infty$ relative error',
               'Suboptimality gap',
               "Optimal nstates"]
    i_error = 2
    error = plot_errors[i_error]
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
    for algo in algos:
        if algo in ["qlearning", "qlearning_es"]:
            flag_ql = True
        elif algo in ["qlearning_ucb", "qlearning_es_ucb"]:
            flag_ucb = True

    for i_algo, algo in enumerate(algos):
        print("algorithm:", algo)
        for i_gamma, gamma in enumerate(gammas):
            print("gamma:", gamma)
            env_name = env_type + str(gammas[i_gamma])
            q_optimal, policy_optimal = q_value_iteration(env, gamma=gamma)
            plot_results(env_name=env_name,
                         algo=algo,
                         gamma=gamma,
                         n_runs=n_runs,
                         error=error,
                         plot_log_value=plot_log_value,
                         q_optimal=q_optimal,
                         policy_optimal=policy_optimal)

    if flag_ql and flag_ucb:
        if plot_log_value:
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-4.5, 2.5)
        elif set_yscale:
            if error == "infinorm":
                ylims = (10 ** (-2), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-2), 10 ** 2)
        elif set_semilog:
            if error == "infinorm":
                ylims = (10 ** (-2), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.6), 10 ** 1.3)
        else:
            if error == "onenorm":
                ylims = (0, 40)
            elif error == "infinorm":
                ylims = (0, 12)
            elif error == "relainfinorm":
                ylims = (0, 4)
            elif error == "gap":
                # ylims = (-0.5, 0.3)
                ylims = (-0.6, 0.25)
            elif error == "nstates_optimal":
                ylims = (0, 6)
    elif flag_ql:
        if plot_log_value:
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-4.5, 1.5)
        elif set_yscale:
            if error == "infinorm":
                ylims = (10 ** (-2), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-2), 10 ** 2)
        elif set_semilog:
            if error == "infinorm":
                ylims = (10 ** (-2), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.6), 10 ** 1.3)
        else:
            if error == "onenorm":
                ylims = (0, 70)
            elif error == "infinorm":
                ylims = (0, 12)
            elif error == "relainfinorm":
                ylims = (0, 4)
            elif error == "gap":
                ylims = (-0.6, 0.25)
            elif error == "nstates_optimal":
                ylims = (0, 6)

    plt.ylim(ylims[0], ylims[1])
    plt.xlim(0, T)
    plt.ylabel(ylabels[i_error], fontsize=14)
    plt.xlabel('Time steps', fontsize=14)
    plt.title("RiverSwim", fontsize=15)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)

    fig_save_directory = "./results/plots/{}/".format("different_gammas")
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

    draw_type = ""
    if flag_ql:
        draw_type += "_QL"
    if flag_ucb:
        draw_type += "_UCB"
    if plot_log_value:
        draw_type += "_log"
    if set_semilog:
        draw_type += "_semilogy"
    if set_yscale:
        draw_type += "_yscale"
    fig_name = env_type + draw_type + note
    plt.savefig(fig_save_directory + fig_name + ".pdf")

    print("Finish plotting.")
