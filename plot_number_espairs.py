import matplotlib.pyplot as plt
import seaborn as sns
import os

from environments.riverswim import RiverswimEnv
from utils import q_value_iteration, plot_mean_CI
from plot import extract_data

sns.set_style("darkgrid")


def plot_settings(env_name, algo):
    if algo == "qlearning":
        linestyle = "-"
        legend = "QL"
        color = "tab:blue"
        marker = None
    elif algo == "qlearning_es":
        linestyle = "--"
        legend = "QL-ES"
        color = "tab:green"
        marker = None
    elif algo == "qlearning_ucb":
        linestyle = "-"
        legend = "UCBQ"
        color = "tab:orange"
        marker = None
    elif algo == "qlearning_es_ucb":
        linestyle = "-."
        legend = "UCBQ-ES"
        color = "tab:red"
        marker = None

    if env_name.split("_")[0] == "riverswim6espairs":
        if algo == "qlearning_es_few1":
            linestyle = "--"
            legend = "QLES(1)"
            color = "tab:brown"
            marker = "."
        elif algo == "qlearning_es_few2":
            linestyle = "--"
            legend = "QLES(2)"
            color = "tab:brown"
            marker = "*"
        elif algo == "qlearning_es_few3":
            linestyle = "--"
            legend = "QLES(3)"
            color = "tab:brown"
            marker = "o"
        elif algo == "qlearning_es_few4":
            linestyle = "--"
            legend = "QLES(4)"
            color = "tab:brown"
            marker = "v"
        elif algo == "qlearning_es_ucb_few1":
            linestyle = "-."
            legend = "UCBQES(1)"
            color = "tab:purple"
            marker = "."
        elif algo == "qlearning_es_ucb_few2":
            linestyle = "-."
            legend = "UCBQES(2)"
            color = "tab:purple"
            marker = "*"
        elif algo == "qlearning_es_ucb_few3":
            linestyle = "-."
            legend = "UCBQES(3)"
            color = "tab:purple"
            marker = "o"
        elif algo == "qlearning_es_ucb_few4":
            linestyle = "-."
            legend = "UCBQES(4)"
            color = "tab:purple"
            marker = "v"
    elif env_name.split("_")[0] == "riverswim20espairs":
        if algo == "qlearning_es_few3":
            linestyle = "--"
            legend = "QLES(3)"
            color = "tab:brown"
            marker = "."
        elif algo == "qlearning_es_few4":
            linestyle = "--"
            legend = "QLES(4)"
            color = "tab:brown"
            marker = "D"
        elif algo == "qlearning_es_few5":
            linestyle = "--"
            legend = "QLES(5)"
            color = "tab:brown"
            marker = "s"
        elif algo == "qlearning_es_few6":
            linestyle = "--"
            legend = "QLES(6)"
            color = "tab:brown"
            marker = "*"
        elif algo == "qlearning_es_few7":
            linestyle = "--"
            legend = "QLES(7)"
            color = "tab:brown"
            marker = "p"
        elif algo == "qlearning_es_few8":
            linestyle = "--"
            legend = "QLES(8)"
            color = "tab:brown"
            marker = "X"
        elif algo == "qlearning_es_few10":
            linestyle = "--"
            legend = "QLES(10)"
            color = "tab:brown"
            marker = "o"
        elif algo == "qlearning_es_few12":
            linestyle = "--"
            legend = "QLES(12)"
            color = "tab:brown"
            marker = "v"
        elif algo == "qlearning_es_few16":
            linestyle = "--"
            legend = "QLES(16)"
            color = "tab:brown"
            marker = "^"
        elif algo == "qlearning_es_ucb_few3":
            linestyle = "-."
            legend = "UCBQES(3)"
            color = "tab:purple"
            marker = "D"
        elif algo == "qlearning_es_ucb_few4":
            linestyle = "-."
            legend = "UCBQES(4)"
            color = "tab:purple"
            marker = "s"
        elif algo == "qlearning_es_ucb_few5":
            linestyle = "-."
            legend = "UCBQES(5)"
            color = "tab:purple"
            marker = "."
        elif algo == "qlearning_es_ucb_few6":
            linestyle = "-."
            legend = "UCBQES(6)"
            color = "tab:purple"
            marker = "*"
        elif algo == "qlearning_es_ucb_few7":
            linestyle = "-."
            legend = "UCBQES(7)"
            color = "tab:purple"
            marker = "p"
        elif algo == "qlearning_es_ucb_few8":
            linestyle = "-."
            legend = "UCBQES(8)"
            color = "tab:purple"
            marker = "X"
        elif algo == "qlearning_es_ucb_few10":
            linestyle = "-."
            legend = "UCBQES(10)"
            color = "tab:purple"
            marker = "o"
        elif algo == "qlearning_es_ucb_few12":
            linestyle = "-."
            legend = "UCBQES(12)"
            color = "tab:purple"
            marker = "v"
        elif algo == "qlearning_es_ucb_few16":
            linestyle = "-."
            legend = "UCBQES(16)"
            color = "tab:purple"
            marker = "^"

    return linestyle, legend, color, marker


if __name__ == '__main__':
    env = RiverswimEnv(nS=6)

    env_type = "riverswim{}espairs_{}".format(env.nS, 2)

    gamma = 0.85
    q_optimal, policy_optimal = q_value_iteration(env, gamma=gamma)

    if env.nS == 6:
        T = 100000
    elif env.nS == 20:
        T = 400000

    # remove qlearning: too bad to plot
    if env.nS == 6:
        algos = [
            # "qlearning",
            "qlearning_ucb",
            # "qlearning_es",
            "qlearning_es_ucb",
            # "qlearning_es_few1",
            # "qlearning_es_few2",
            # "qlearning_es_few3",
            # "qlearning_es_few4",
            "qlearning_es_ucb_few1",
            "qlearning_es_ucb_few2",
            "qlearning_es_ucb_few3",
            "qlearning_es_ucb_few4",
        ]
    elif env.nS == 20:
        algos = [
            # "qlearning",
            # "qlearning_ucb",
            "qlearning_es",
            # "qlearning_es_ucb",
            "qlearning_es_few3",
            # "qlearning_es_few4",
            # "qlearning_es_few5",
            "qlearning_es_few6",
            # "qlearning_es_few7",
            # "qlearning_es_few8",
            "qlearning_es_few10",
            "qlearning_es_few12",
            # "qlearning_es_few16",
            # "qlearning_es_ucb_few3",
            # "qlearning_es_ucb_few4",
            # "qlearning_es_ucb_few5",
            # "qlearning_es_ucb_few6",
            # "qlearning_es_ucb_few7",
            # "qlearning_es_ucb_few8",
            # "qlearning_es_ucb_few10",
            # "qlearning_es_ucb_few12",
            # "qlearning_es_ucb_few16"
        ]

    n_runs = 100
    c_lr = 4.0
    # error types: onenorm, infinorm, relainfinorm, gap, nstates_optimal
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
    i_error = 5
    error = plot_errors[i_error]
    print("env name:", env_type)
    print("error type:", error)

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
        env_name = env_type
        plot_values = extract_data(n_runs, env_name, algo, error, plot_log_value, q_optimal, policy_optimal)
        linestyle, legend, color, marker = plot_settings(env_name, algo)
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

    # plot all or only some es pairs
    plot_few = True

    if env_type.split("_")[0] == "riverswim6espairs":
        if plot_log_value:
            if error == "infinorm":
                ylims = (-4.5, 0)
            elif error == "relainfinorm":
                ylims = (-4.5, 1)
        elif set_yscale:
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.05), 10 ** 1.05)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-1.05), 10 ** 1.05)
                else:
                    ylims = (10 ** (-1.05), 10 ** 1.05)
                # ylims = (10 ** (-1.8), 10 ** 0)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.8), 10 ** 0.3)
        elif set_semilog:
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.05), 10 ** 1.05)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-1.05), 10 ** 1.05)
                else:
                    ylims = (10 ** (-1.05), 10 ** 1.05)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 0.5)
        else:
            if error == "onenorm":
                ylims = (0, 6)
            elif error == "infinorm":
                ylims = (0, 0.75)
            elif error == "relainfinorm":
                ylims = (0, 2)
            elif error == "gap":
                if c_lr == 4.0:
                    if (flag_ql == True) and (flag_ucb == False):
                        # ylims = (-0.4, 0.25)  # with ql
                        ylims = (-0.2, 0.25)  # without ql
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (-0.15, 0.05)
                    else:
                        ylims = (-0.4, 0.25)
                else:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (-0.1, 0.15)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (-0.1, 0.15)
                    else:
                        ylims = (-0.1, 0.15)
            elif error == "nstates_optimal":
                ylims = (4.9, 6)
            elif error == "nstates_suboptimal":
                if c_lr == 4.0:
                    if (flag_ql == True) and (flag_ucb == False):
                        # ylims = (0, 7)
                        ylims = (0, 4)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (0, 1.61)
                    else:
                        ylims = (0, 7)
                else:
                    ylims = (0, 7)
    elif env_type.split("_")[0] == "riverswim20espairs":
        if plot_few:
            if plot_log_value:
                if error == "infinorm":
                    ylims = (-2.5, 0)
                elif error == "relainfinorm":
                    ylims = (-2, 3)
            elif set_yscale:
                if error == "infinorm":
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.05), 10 ** 0.8)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.05), 10 ** 0.4)
                    else:
                        ylims = (10 ** (-1.05), 10 ** 0.1)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1), 10 ** 1.2)
            elif set_semilog:
                if error == "infinorm":
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.05), 10 ** 0.8)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.05), 10 ** 0.4)
                    else:
                        ylims = (10 ** (-1.05), 10 ** 0.1)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.5), 10 ** 1.2)
            else:
                if error == "onenorm":
                    ylims = (0, 15)
                elif error == "infinorm":
                    ylims = (0, 1)
                elif error == "relainfinorm":
                    ylims = (0, 12)
                elif error == "gap":
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (-0.04, 0.02)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (-0.04, 0.015)
                    else:
                        ylims = (-0.08, 0.01)
                elif error == "nstates_optimal":
                    ylims = (0, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 10)
        else:
            if plot_log_value:
                if error == "infinorm":
                    ylims = (-2.5, 0.5)
                elif error == "relainfinorm":
                    ylims = (-2, 3.5)
            elif set_yscale:
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 0.5)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1), 10 ** 1.5)
            elif set_semilog:
                if error == "infinorm":
                    ylims = (10 ** (-1), 10 ** 0.5)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.3), 10 ** 1.5)
            else:
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 2)
                elif error == "relainfinorm":
                    ylims = (0, 25)
                elif error == "gap":
                    ylims = (-0.3, 0.01)
                elif error == "nstates_optimal":
                    ylims = (11, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 20)

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
