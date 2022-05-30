import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from environments.riverswim import RiverswimEnv
from utils import q_value_iteration, load_obj, plot_mean_CI
from plot import extract_data

sns.set_style('darkgrid')


def plot_results(env_name,
                 algo,
                 n_runs,
                 error,
                 plot_log_value,
                 q_optimal,
                 policy_optimal):
    # all colors: tab:blue, tab:orange, tab:green, tab:red, tab:purple
    #             tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan

    # all line styles: -, --, -., :

    # makers: ".", "*", "x", "1", "|", "v", "d"

    plot_values = extract_data(n_runs, env_name, algo, error, plot_log_value, q_optimal, policy_optimal)

    use_few_partitions = True

    if algo == "qlearning":
        legend = "QL"
        linestyle = "-"
        color = "tab:blue"
        marker = None
    elif algo == "qlearning_ucb":
        legend = "UCBQ"
        linestyle = "-"
        color = "tab:orange"
        marker = None
    elif algo == "qlearning_es":
        legend = "QL-ES"
        linestyle = "--"
        color = "tab:green"
        marker = None
    elif algo == "qles_partitionC1":
        legend = "QLC1"
        linestyle = "--"
        color = "tab:brown"
        marker = "."
    elif algo == "qles_partitionC2":
        legend = "QLC2"
        linestyle = "--"
        color = "tab:brown"
        marker = "p"
    elif algo == "qles_partitionC3":
        if use_few_partitions:
            legend = "QLC1"
        else:
            legend = "QLC3"
        linestyle = "--"
        color = "tab:brown"
        marker = "*"
    elif algo == "qles_partitionC4":
        if use_few_partitions:
            legend = "QLC2"
        else:
            legend = "QLC4"
        linestyle = "--"
        color = "tab:brown"
        marker = "^"
    elif algo == "qlearning_es_ucb":
        legend = "UCBQ-ES"
        linestyle = "-."
        color = "tab:red"
        marker = None
    elif algo == "ucbqes_partitionC1":
        legend = "UCBQC1"
        linestyle = "-."
        color = "tab:purple"
        marker = "."
    elif algo == "ucbqes_partitionC2":
        legend = "UCBQC2"
        linestyle = "-."
        color = "tab:purple"
        marker = "p"
    elif algo == "ucbqes_partitionC3":
        if use_few_partitions:
            legend = "UCBQC1"
        else:
            legend = "UCBQC3"
        linestyle = "-."
        color = "tab:purple"
        marker = "*"
    elif algo == "ucbqes_partitionC4":
        if use_few_partitions:
            legend = "UCBQC2"
        else:
            legend = "UCBQC4"
        linestyle = "-."
        color = "tab:purple"
        marker = "^"

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
                 marker_every=10)


if __name__ == '__main__':
    env = RiverswimEnv(nS=6)

    env_type = "riverswim{}partition".format(env.nS)

    gamma = 0.85
    q_optimal, policy_optimal = q_value_iteration(env)

    if env.nS == 6:
        T = 100000
    elif env.nS == 8:
        T = 120000

    algos = ["qlearning",
             "qlearning_ucb",
             "qlearning_es",
             # "qles_partitionC1",
             # "qles_partitionC2",
             "qles_partitionC3",
             "qles_partitionC4",
             "qlearning_es_ucb",
             # "ucbqes_partitionC1",
             # "ucbqes_partitionC2",
             "ucbqes_partitionC3",
             "ucbqes_partitionC4"]

    n_runs = 100
    # error types: onenorm, infinorm, relainfinorm, gap
    plot_errors = ["onenorm", "infinorm", "relainfinorm", "gap", "nstates_optimal"]
    ylabels = ['$\ell_1$ absolute error',
               '$\ell_\infty$ absolute error',
               '$\ell_\infty$ relative error',
               'Suboptimality gap',
               "Optimal nstates"]
    i_error = 0
    error = plot_errors[i_error]
    print("env name:", env_type)
    print("error type:", plot_errors[i_error])

    plt.figure(figsize=(8, 6))
    plt.tight_layout()

    # plot settings
    set_ci = True  # first step, second step, third step
    plot_log_value = False  # second step
    set_yscale = False  # third step
    set_semilog = False  # fourth step

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
        plot_results(env_name=env_type,
                     algo=algo,
                     n_runs=n_runs,
                     error=error,
                     plot_log_value=plot_log_value,
                     q_optimal=q_optimal,
                     policy_optimal=policy_optimal)

    if env.nS == 6:
        if flag_ql and flag_ucb:
            if plot_log_value:
                if error == "infinorm":
                    ylims = (-4.3, 2.1)
                elif error == "relainfinorm":
                    ylims = (-4.5, 1.2)
            elif set_yscale:
                if error == "infinorm":
                    ylims = (10 ** (-1.8), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.8), 10 ** 1.5)
            elif set_semilog:
                if error == "infinorm":
                    ylims = (10 ** (-1.7), 10 ** 1.7)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.4), 10 ** 1.2)
            else:
                if error == "onenorm":
                    ylims = (0, 45)
                    # ylims = (0, 25)
                elif error == "infinorm":
                    ylims = (0, 15)
                    # ylims = (0, 6)
                elif error == "relainfinorm":
                    ylims = (0, 4)
                elif error == "gap":
                    ylims = (-0.6, 0.1)
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
    elif env.nS == 8:
        if flag_ql and flag_ucb:
            if plot_log_value:
                if error == "infinorm":
                    ylims = (-4, 2)
                elif error == "relainfinorm":
                    ylims = (-3.5, 2)
            elif set_yscale:
                if error == "infinorm":
                    ylims = (10 ** (-1.5), 10 ** 1.2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.1)
            elif set_semilog:
                if error == "infinorm":
                    ylims = (10 ** (-1.3), 10 ** 1)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1), 10 ** 1)
            else:
                if error == "onenorm":
                    ylims = (0, 25)
                elif error == "infinorm":
                    ylims = (0, 8)
                elif error == "relainfinorm":
                    ylims = (0, 4)
                elif error == "gap":
                    ylims = (-0.6, 0.1)
                elif error == "nstates_optimal":
                    ylims = (0, 8)
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
