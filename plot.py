import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environments.riverswim import (
    RiverswimEnv,
    RiverswimEnvKappa1,
    RiverswimEnvKappa2,
    RiverswimEnvKappa3
)
from environments.gridworld import GridworldEnv
from utils import q_value_iteration, load_obj, plot_mean_CI

use_seaborn = True
if use_seaborn:
    sns.set_style('darkgrid')


def plot_title_ylims(env_name,
                     error,
                     use_kappaenv,
                     plot_log_value,
                     set_semilog,
                     set_yscale,
                     flag_ql,
                     flag_ucb,
                     limit_q_value,
                     c_lr):
    if env_name.split("_")[0] == "riverswim6":
        T = 100000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** (-2), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.7), 10 ** 1.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** (-1.7), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 1.5)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 4)
            elif error == "relainfinorm":
                ylims = (-5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 26)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 6)
            elif error == "gap":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (-0.35, 0.15)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (-0.25, 0.1)
                else:
                    ylims = (-0.4, 0.25)
            elif error == "nstates_optimal":
                ylims = (0, 6)
            elif error == "nstates_suboptimal":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (0, 7)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (0, 6)
                else:
                    ylims = (0, 7)
        plt_title = "RiverSwim"
    elif env_name.split("_")[0] == "riverswim6kappa1":
        T = 100000
        if use_kappaenv:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.5), 10 ** 1.5)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-0.5), 10 ** 1.5)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.3), 10 ** 1.5)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-1.5, 2.5)
                elif error == "relainfinorm":
                    ylims = (-1.2, 1)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 15)
                elif error == "relainfinorm":
                    ylims = (0, 12)
                elif error == "gap":
                    # ylims = (-0.5, 0.25)
                    ylims = (-2.5, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 6)
                elif error == "nstates_suboptimal":
                    ylims = (0, 6)
        else:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.5), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.5), 10 ** 1.8)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.5), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.1), 10 ** 1.8)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3.2, 3)
                elif error == "relainfinorm":
                    ylims = (-3, 2)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 15)
                elif error == "relainfinorm":
                    ylims = (0, 11)
                elif error == "gap":
                    # ylims = (-0.5, 0.25)
                    ylims = (-2.5, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 6)
                elif error == "nstates_suboptimal":
                    ylims = (0, 6)
        plt_title = "RiverSwim, $\epsilon=0.1$"
    elif env_name.split("_")[0] == "riverswim6kappa2":
        T = 100000
        if use_kappaenv:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-0.8), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.5), 10 ** 1.5)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-0.8), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.5), 10 ** 1.5)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-1.5, 2.5)
                elif error == "relainfinorm":
                    ylims = (-1, 1)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 15)
                elif error == "relainfinorm":
                    ylims = (0, 12)
                elif error == "gap":
                    ylims = (-1.5, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 6)
                elif error == "nstates_suboptimal":
                    ylims = (0, 6)
        else:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.7), 10 ** 2.2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.5), 10 ** 2)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.3), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1), 10 ** 1.7)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3.5, 3)
                elif error == "relainfinorm":
                    ylims = (-3.5, 1.7)
            else:
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 14)
                elif error == "relainfinorm":
                    ylims = (0, 12)
                elif error == "gap":
                    ylims = (-1.5, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 6)
                elif error == "nstates_suboptimal":
                    ylims = (0, 6)
        plt_title = "RiverSwim, $\epsilon=0.3$"
    elif env_name.split("_")[0] == "riverswim6kappa3":
        T = 100000
        if use_kappaenv:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-0.8), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.5), 10 ** 1.5)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-0.8), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.5), 10 ** 1.5)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-1.5, 2.5)
                elif error == "relainfinorm":
                    ylims = (-1, 1)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 15)
                elif error == "relainfinorm":
                    ylims = (0, 12)
                elif error == "gap":
                    ylims = (-2, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 6)
                elif error == "nstates_suboptimal":
                    ylims = (0, 6)
        else:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.5), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.5), 10 ** 2)
            elif set_semilog:
                print("semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.3), 10 ** 2)
                elif error == "relainfinorm":
                    ylims = (10 ** (-1.1), 10 ** 1.7)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3, 3)
                elif error == "relainfinorm":
                    ylims = (-3.2, 1.7)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 15)
                elif error == "relainfinorm":
                    ylims = (0, 10)
                elif error == "gap":
                    ylims = (-2, 0.25)
                elif error == "nstates_optimal":
                    ylims = (0, 6)
                elif error == "nstates_suboptimal":
                    ylims = (0, 6)
        plt_title = "RiverSwim, $\epsilon=0.3$"
    elif env_name == "riverswim10":
        T = 200000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** -1.7, 10 ** 1.7)
            elif error == "relainfinorm":
                ylims = (10 ** -1.3, 10 ** 1.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** -1.5, 10 ** 1.5)
            elif error == "relainfinorm":
                ylims = (10 ** -1.1, 10 ** 1.7)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-4.5, 4)
            elif error == "relainfinorm":
                ylims = (-3.5, 3.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 35)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 15)
            elif error == "gap":
                # ylims = (-3.0, 0.1)
                ylims = (-0.5, 0.1)
            elif error == "nstates_optimal":
                ylims = (0, 10)
            elif error == "nstates_suboptimal":
                ylims = (0, 10)
        plt_title = "RiverSwim"
    elif env_name.split("_")[0] == "riverswim20":
        T = 400000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-0.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-1.1), 10 ** 0.5)
                else:
                    ylims = (10 ** (-1.2), 10 ** 1.1)
            elif error == "relainfinorm":
                ylims = (10 ** (-1), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if c_lr == 1.0:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-0.1), 10 ** 1.1)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.1), 10 ** 0.5)
                    else:
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                else:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.1), 10 ** 0.5)
                    else:
                        ylims = (10 ** (-1.2), 10 ** 1.1)
            elif error == "relainfinorm":
                ylims = (10 ** (-0.2), 10 ** 2)
        elif plot_log_value:
            print("scale: data log")
            if error == "onenorm":
                ylims = (-3, 5)
            elif error == "infinorm":
                ylims = (-3, 3.5)
            elif error == "relainfinorm":
                ylims = (-2, 4.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 40)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 35)
            elif error == "gap":
                if limit_q_value:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (-0.1, 0.03)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (-0.06, 0.01)
                    else:
                        ylims = (-0.1, 0.03)
                else:
                    if c_lr == 1.0:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (-0.01, 0.0026)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (-0.06, 0.01)
                        else:
                            ylims = (-0.06, 0.02)
                    else:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (-0.05, 0.01)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (-0.06, 0.01)
                        else:
                            ylims = (-0.06, 0.02)
            elif error == "nstates_optimal":
                ylims = (0, 20)
            elif error == "nstates_suboptimal":
                ylims = (0, 20)
        plt_title = "RiverSwim"
    elif env_name.split("_")[0] == "riverswim20kappa1":
        T = 400000
        if use_kappaenv:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    # ylims = (10 ** (-1.2), 10 ** 1.8)
                    if c_lr == 1.0:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                        else:
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                    elif c_lr == 3.0:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                        else:
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                    else:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                        else:
                            ylims = (10 ** (-1.05), 10 ** 1.1)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.6), 10 ** 2)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.05), 10 ** 1.1)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.05), 10 ** 1.1)
                    else:
                        ylims = (10 ** (-1.05), 10 ** 1.1)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.6), 10 ** 1.8)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3, 3.5)
                elif error == "relainfinorm":
                    ylims = (-1.2, 4)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 7)
                elif error == "relainfinorm":
                    ylims = (0, 20)
                elif error == "gap":
                    if c_lr == 1.0:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (-0.015, 0.005)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (-0.08, 0.02)
                        else:
                            ylims = (-0.08, 0.02)
                    elif c_lr == 3.0:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (-0.04, 0.02)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (-0.08, 0.02)
                        else:
                            ylims = (-0.08, 0.02)
                    else:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (-0.15, 0.02)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (-0.08, 0.02)
                        else:
                            ylims = (-0.15, 0.02)
                elif error == "nstates_optimal":
                    ylims = (0, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 20)
        else:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (0), 10 ** 2)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.5)
                elif error == "relainfinorm":
                    ylims = (10 ** (0), 10 ** 1.8)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3, 3.5)
                elif error == "relainfinorm":
                    ylims = (-0.2, 4)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 7)
                elif error == "relainfinorm":
                    ylims = (0, 20)
                elif error == "gap":
                    ylims = (-0.25, 0.05)
                elif error == "nstates_optimal":
                    ylims = (0, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 20)
        plt_title = "RiverSwim, $\epsilon=0.1$"
    elif env_name == "riverswim20kappa2":
        T = 400000
        if use_kappaenv:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.6), 10 ** 2)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.3)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.6), 10 ** 1.8)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3, 3.5)
                elif error == "relainfinorm":
                    ylims = (-1.5, 4)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 7)
                elif error == "relainfinorm":
                    ylims = (0, 20)
                elif error == "gap":
                    ylims = (-0.25, 0.05)
                elif error == "nstates_optimal":
                    ylims = (0, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 20)
        else:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (0), 10 ** 2)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.5)
                elif error == "relainfinorm":
                    ylims = (10 ** (0), 10 ** 1.8)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3, 3.5)
                elif error == "relainfinorm":
                    ylims = (-0.2, 4)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 7)
                elif error == "relainfinorm":
                    ylims = (0, 20)
                elif error == "gap":
                    ylims = (-0.25, 0.05)
                elif error == "nstates_optimal":
                    ylims = (0, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 20)
        plt_title = "RiverSwim, $\epsilon=0.3$"
    elif env_name == "riverswim20kappa3":
        T = 400000
        if use_kappaenv:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.6), 10 ** 2.2)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.6)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.6), 10 ** 2)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3, 3.5)
                elif error == "relainfinorm":
                    ylims = (-1.5, 4)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 10)
                elif error == "relainfinorm":
                    ylims = (0, 30)
                elif error == "gap":
                    ylims = (-0.25, 0.05)
                elif error == "nstates_optimal":
                    ylims = (0, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 20)
        else:
            if set_yscale:
                print("scale: yaxis log")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.8)
                elif error == "relainfinorm":
                    ylims = (10 ** (-0.1), 10 ** 2.2)
            elif set_semilog:
                print("scale: semilogy")
                if error == "infinorm":
                    ylims = (10 ** (-1.2), 10 ** 1.6)
                elif error == "relainfinorm":
                    ylims = (10 ** (0), 10 ** 2)
            elif plot_log_value:
                print("scale: data log")
                if error == "infinorm":
                    ylims = (-3, 3.5)
                elif error == "relainfinorm":
                    ylims = (-0.3, 4)
            else:
                print("scale: linear")
                if error == "onenorm":
                    ylims = (0, 40)
                elif error == "infinorm":
                    ylims = (0, 10)
                elif error == "relainfinorm":
                    ylims = (0, 30)
                elif error == "gap":
                    ylims = (-0.25, 0.05)
                elif error == "nstates_optimal":
                    ylims = (0, 20)
                elif error == "nstates_suboptimal":
                    ylims = (0, 20)
        plt_title = "RiverSwim, $\epsilon=0.3$"
    elif env_name == "riverswim25":
        T = 500000
        if set_yscale:
            print("scale: yaxis log")
            ylims = (10 ** -2, 10 ** 2)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-3, 3.5)
            elif error == "relainfinorm":
                ylims = (-2, 5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 35)
            elif error == "gap":
                ylims = (-0.4, 0.25)
        plt_title = "RiverSwim"
    elif env_name == "riverswim40":
        T = 800000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** (-1.2), 10 ** 1.8)
            elif error == "relainfinorm":
                ylims = (10 ** (-0.2), 10 ** 3.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** (-1.2), 10 ** 1.7)
            elif error == "relainfinorm":
                ylims = (10 ** 0, 10 ** 3.2)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-3.5, 3.5)
            elif error == "relainfinorm":
                ylims = (-0.5, 8)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 60)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 200)
            elif error == "gap":
                ylims = (-0.6, 0.2)
            elif error == "nstates_optimal":
                ylims = (0, 40)
            elif error == "nstates_suboptimal":
                ylims = (0, 40)
        plt_title = "RiverSwim"
    elif env_name.split("_")[0] == "riverswim50":
        T = 1000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                # ylims = (10 ** (-1.2), 10 ** 1.8)
                if c_lr in [1.0, 2.0]:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                    else:
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                else:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.3), 10 ** 1.1)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.3), 10 ** 1.1)
                    else:
                        ylims = (10 ** (-1.3), 10 ** 1.1)
            elif error == "relainfinorm":
                ylims = (10 ** (-1), 10 ** 5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if c_lr in [1.0, 2.0]:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                    else:
                        ylims = (10 ** (-1.2), 10 ** 1.1)
                else:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (10 ** (-1.3), 10 ** 1.1)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (10 ** (-1.1), 10 ** 1)
                    else:
                        ylims = (10 ** (-1.3), 10 ** 1.1)
            elif error == "relainfinorm":
                ylims = (10 ** 0.5, 10 ** 4.5)
        elif plot_log_value:
            print("scale: data log")
            if error == "onenorm":
                ylims = (-3, 5)
            elif error == "infinorm":
                ylims = (-3.5, 3.5)
            elif error == "relainfinorm":
                ylims = (-0.5, 9.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 100)
            elif error == "infinorm":
                # ylims = (0, 15) # 50 runs
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 1300)
            elif error == "gap":
                if limit_q_value:
                    if (flag_ql == True) and (flag_ucb == False):
                        ylims = (-0.08, 0.02)
                    elif (flag_ql == False) and (flag_ucb == True):
                        ylims = (-0.05, 0.01)
                    else:
                        ylims = (-0.08, 0.02)
                else:
                    if c_lr in [1.0, 2.0]:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (-0.005, 0.005)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (-0.05, 0.01)
                        else:
                            ylims = (-0.08, 0.02)
                    else:
                        if (flag_ql == True) and (flag_ucb == False):
                            ylims = (-0.08, 0.02)
                        elif (flag_ql == False) and (flag_ucb == True):
                            ylims = (-0.05, 0.01)
                        else:
                            ylims = (-0.08, 0.02)
            elif error == "nstates_optimal":
                ylims = (0, 50)
            elif error == "nstates_suboptimal":
                ylims = (0, 50)
        plt_title = "RiverSwim"
    elif env_name[0:-2] == "gridworld2room_7_7":
        # T = 2000000
        T = 1000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) or (flag_ucb == False):
                    ylims = (10 ** (-2.1), 10 ** 1.1)
                elif (flag_ql == False) or (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 2)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if error == "infinorm":
                    if (flag_ql == True) or (flag_ucb == False):
                        ylims = (10 ** (-2.1), 10 ** 1.1)
                    elif (flag_ql == False) or (flag_ucb == True):
                        ylims = (10 ** (-2.5), 10 ** 2.5)
                    else:
                        ylims = (10 ** (-2.5), 10 ** 2.5)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) or (flag_ucb == False):
                    ylims = (-0.13, 0.02)
                elif (flag_ql == False) or (flag_ucb == True):
                    ylims = (-0.002, 0.0008)
                else:
                    ylims = (-0.002, 0.0008)
            elif error == "nstates_optimal":
                ylims = (0, 21)
            elif error == "nstates_suboptimal":
                ylims = (0, 21)
        plt_title = "GridWorld"
    elif (env_name == "gridworld2room_7_7_kappa1") or (env_name[0:-2] == "gridworld2room_7_7_kappa1"):
        T = 2000000
        # T = 1000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-2.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 2)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-2.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.1), 10 ** 0.1)
                else:
                    ylims = (10 ** (-2), 10 ** 3)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (-0.1, 0.02)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (-0.13, 0.01)
                else:
                    ylims = (-0.13, 0.01)
            elif error == "nstates_optimal":
                ylims = (0, 21)
            elif error == "nstates_suboptimal":
                ylims = (0, 21)
        plt_title = "GridWorld, $\epsilon=0.1$"
    elif env_name == "gridworld2room_9_9":
        T = 2000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2.5)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                ylims = (-0.002, 0.0008)
            elif error == "nstates_optimal":
                ylims = (0, 43)
            elif error == "nstates_suboptimal":
                ylims = (0, 43)
        plt_title = "GridWorld"
    elif env_name[0:-2] == "gridworld2room_9_9_kappa1":
        T = 1000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-2.5), 10 ** 2)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.1), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-2.5), 10 ** 2.5)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.1), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2.5)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (-0.002, 0.0008)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (-0.02, 0.005)
                else:
                    ylims = (-0.002, 0.0008)
            elif error == "nstates_optimal":
                ylims = (0, 43)
            elif error == "nstates_suboptimal":
                ylims = (0, 43)
        plt_title = "GridWorld, $\epsilon=0.1$"
    elif (env_name == "gridworld2room_11_11") or (env_name[0:-2] == "gridworld2room_11_11"):
        T = 2000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-2.5), 10 ** 2)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-2.5), 10 ** 2, 5)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2.5)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (-0.002, 0.0008)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (-0.015, 0.005)
                else:
                    ylims = (-0.002, 0.0008)
            elif error == "nstates_optimal":
                ylims = (0, 73)
            elif error == "nstates_suboptimal":
                ylims = (0, 73)
        plt_title = "GridWorld"
    elif env_name == "gridworld2room_11_11_kappa1":
        T = 2000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2.5)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                ylims = (-0.002, 0.0008)
            elif error == "nstates_optimal":
                ylims = (0, 73)
            elif error == "nstates_suboptimal":
                ylims = (0, 73)
        plt_title = "GridWorld"
    elif env_name == "gridworld2room_13_13":
        T = 2000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2.5)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                ylims = (-0.002, 0.0008)
            elif error == "nstates_optimal":
                ylims = (0, 111)
            elif error == "nstates_suboptimal":
                ylims = (0, 111)
        plt_title = "GridWorld"
    elif env_name == "gridworld2room_13_13_kappa1":
        T = 2000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2.5)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 3)
        elif plot_log_value:
            print("scale: data log")
            if error == "infinorm":
                ylims = (-5, 3)
            elif error == "relainfinorm":
                ylims = (-3.5, 3)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 50)
            elif error == "infinorm":
                ylims = (0, 20)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                ylims = (-0.002, 0.0008)
            elif error == "nstates_optimal":
                ylims = (0, 111)
            elif error == "nstates_suboptimal":
                ylims = (0, 111)
        plt_title = "GridWorld"
    elif (env_name == "gridworld4room_7_7") or (env_name[0:-2] == "gridworld4room_7_7"):
        # T = 2000000
        T = 1000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) or (flag_ucb == False):
                    ylims = (10 ** (-2.1), 10 ** 1.1)
                elif (flag_ql == False) or (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 2)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if (flag_ql == True) or (flag_ucb == False):
                    ylims = (10 ** (-2.1), 10 ** 1.1)
                elif (flag_ql == False) or (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 2)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif plot_log_value:
            print("scale: data log")
            if error == "onenorm":
                ylims = (-3, 5)
            elif error == "infinorm":
                ylims = (-5, 2)
            elif error == "relainfinorm":
                ylims = (-3.5, 2.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 70)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) or (flag_ucb == False):
                    ylims = (-0.09, 0.02)
                elif (flag_ql == False) or (flag_ucb == True):
                    ylims = (-0.015, 0.006)
                else:
                    ylims = (-0.015, 0.006)
            elif error == "nstates_optimal":
                ylims = (0, 20)
            elif error == "nstates_suboptimal":
                ylims = (0, 20)
        plt_title = "GridWorld"
    elif (env_name == "gridworld4room_9_9") or (env_name[0:-2] == "gridworld4room_9_9"):
        # T = 4000000
        # T = 2000000
        T = 3000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 4)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 6)
        elif plot_log_value:
            print("scale: data log")
            if error == "onenorm":
                ylims = (-3, 5)
            elif error == "infinorm":
                ylims = (-5, 2)
            elif error == "relainfinorm":
                ylims = (-3.5, 2.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 70)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (-0.06, 0.01)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (-0.0061, 0.0021)
                else:
                    ylims = (-0.015, 0.006)
            elif error == "nstates_optimal":
                ylims = (0, 40)
            elif error == "nstates_suboptimal":
                ylims = (0, 40)
        plt_title = "GridWorld"
    elif env_name == "gridworld4room_9_9_kappa1":
        T = 4000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                ylims = (10 ** (-2.5), 10 ** 4)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 6)
        elif plot_log_value:
            print("scale: data log")
            if error == "onenorm":
                ylims = (-3, 5)
            elif error == "infinorm":
                ylims = (-5, 2)
            elif error == "relainfinorm":
                ylims = (-3.5, 2.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 70)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                ylims = (-0.015, 0.006)
            elif error == "nstates_optimal":
                ylims = (0, 40)
            elif error == "nstates_suboptimal":
                ylims = (0, 40)
        plt_title = "GridWorld"
    elif (env_name == "gridworld4room_11_11") or (env_name[0:-2] == "gridworld4room_11_11"):
        # T = 2000000
        T = 1000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 4)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 6)
        elif plot_log_value:
            print("scale: data log")
            if error == "onenorm":
                ylims = (-3, 5)
            elif error == "infinorm":
                ylims = (-5, 2)
            elif error == "relainfinorm":
                ylims = (-3.5, 2.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 70)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (-0.06, 0.01)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (-0.0061, 0.0021)
                else:
                    ylims = (-0.015, 0.006)
            elif error == "nstates_optimal":
                ylims = (0, 68)
            elif error == "nstates_suboptimal":
                ylims = (0, 68)
        plt_title = "GridWorld"
    elif (env_name == "gridworld4room_13_13") or (env_name[0:-2] == "gridworld4room_13_13"):
        T = 2000000
        if set_yscale:
            print("scale: yaxis log")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 2)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 2.5)
        elif set_semilog:
            print("scale: semilogy")
            if error == "infinorm":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (10 ** (-1.1), 10 ** 1.1)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (10 ** (-2.5), 10 ** 1.1)
                else:
                    ylims = (10 ** (-2.5), 10 ** 4)
            elif error == "relainfinorm":
                ylims = (10 ** (-1.5), 10 ** 6)
        elif plot_log_value:
            print("scale: data log")
            if error == "onenorm":
                ylims = (-3, 5)
            elif error == "infinorm":
                ylims = (-5, 2)
            elif error == "relainfinorm":
                ylims = (-3.5, 2.5)
        else:
            print("scale: linear")
            if error == "onenorm":
                ylims = (0, 70)
            elif error == "infinorm":
                ylims = (0, 10)
            elif error == "relainfinorm":
                ylims = (0, 20)
            elif error == "gap":
                if (flag_ql == True) and (flag_ucb == False):
                    ylims = (-0.06, 0.01)
                elif (flag_ql == False) and (flag_ucb == True):
                    ylims = (-0.0061, 0.0021)
                else:
                    ylims = (-0.015, 0.006)
            elif error == "nstates_optimal":
                ylims = (0, 104)
            elif error == "nstates_suboptimal":
                ylims = (0, 104)
        plt_title = "GridWorld"

    return T, ylims, plt_title


def extract_data(n_runs, env_name, algo, error, plot_log_value, q_optimal, policy_optimal):
    # for each seed
    onenorm_qlearning = []
    infinorm_qlearning = []
    relainfinorm_qlearning = []
    gap_qlearning = []
    nsoptm_qlearning = []
    nssuboptm_qlearning = []
    for i_seed in range(n_runs):
        saved_name = "./results/{}/{}/q_{}".format(env_name, algo, i_seed)
        if env_name in ["riverswim8partition",
                        "riverswim10",
                        "riverswim20",
                        "riverswim20espairs",
                        "riverswim20kappa1",
                        "riverswim20kappa2",
                        "riverswim20kappa3",
                        "riverswim25",
                        "riverswim40",
                        "riverswim50",
                        "gridworld2room_7_7",
                        "gridworld2room_7_7_kappa1",
                        "gridworld4room_7_7",
                        "gridworld4room_9_9"]:
            # remove some high values in large states of riverswim
            if not os.path.exists(saved_name + ".pkl"):
                continue
        q_all = list(load_obj(saved_name).values())

        onenorm_t = []
        infinorm_t = []
        relainfinorm_t = []
        gap_t = []
        nsoptm_t = []
        nssuboptm_t = []
        for q in q_all:
            q_t = np.array(list(q.values()))
            if error == "onenorm":
                onenorm_t.append(np.sum(np.abs(q_optimal - q_t)))
            elif error == "infinorm":
                infinorm_t.append(np.abs(q_optimal - q_t).max())
            elif error == "relainfinorm":
                relainfinorm_t.append(np.abs((q_optimal - q_t) / q_optimal).max())
            elif error == "gap":
                # gap_t.append((q_t[:, 1] - q_t[:, 0]).min())
                # compute Q(s,a*)-Q(s,a)
                h, w = np.shape(q_t)
                mask = np.ones((h, w), dtype=bool)
                mask[range(h), policy_optimal] = False
                # first way
                q_optimal_policy = q_t[~mask].reshape(-1, 1)
                q_table_pop = q_t[mask].reshape(h, w - 1)
                gap_t.append(np.min(q_optimal_policy - q_table_pop))
                # second way = first way
                # q_optimal_policy = q_t[~mask]
                # q_table_pop = q_t[mask].reshape(h, w - 1)
                # q_table_suboptimal = np.max(q_table_pop, axis=1)
                # gap_t.append(np.min(q_optimal_policy - q_table_suboptimal))
            elif error == "nstates_optimal":
                h, w = np.shape(q_t)
                mask = np.ones((h, w), dtype=bool)
                mask[range(h), policy_optimal] = False
                q_optimal_policy = q_t[~mask].reshape(-1, 1)
                q_table_pop = q_t[mask].reshape(h, w - 1)
                nsoptm_t.append(np.count_nonzero(np.min((q_optimal_policy - q_table_pop), axis=1) > 0))
            elif error == "nstates_suboptimal":
                h, w = np.shape(q_t)
                mask = np.ones((h, w), dtype=bool)
                mask[range(h), policy_optimal] = False
                q_optimal_policy = q_t[~mask].reshape(-1, 1)
                q_table_pop = q_t[mask].reshape(h, w - 1)
                nssuboptm_t.append(np.count_nonzero(np.min((q_optimal_policy - q_table_pop), axis=1) <= 0))

        onenorm_qlearning.append(onenorm_t)
        infinorm_qlearning.append(infinorm_t)
        relainfinorm_qlearning.append(relainfinorm_t)
        gap_qlearning.append(gap_t)
        nsoptm_qlearning.append(nsoptm_t)
        nssuboptm_qlearning.append(nssuboptm_t)
        if (i_seed + 1) % 10 == 0:
            print(i_seed)

    if error == "onenorm":
        if plot_log_value:
            plot_values = np.log(np.array(onenorm_qlearning))
        else:
            plot_values = np.array(onenorm_qlearning)
    elif error == "infinorm":
        if plot_log_value:
            plot_values = np.log(np.array(infinorm_qlearning))
        else:
            plot_values = np.array(infinorm_qlearning)
    elif error == "relainfinorm":
        if plot_log_value:
            plot_values = np.log(np.array(relainfinorm_qlearning))
        else:
            plot_values = np.array(relainfinorm_qlearning)
    elif error == "gap":
        plot_values = np.array(gap_qlearning)
    elif error == "nstates_optimal":
        plot_values = np.array(nsoptm_qlearning)
    elif error == "nstates_suboptimal":
        plot_values = np.array(nssuboptm_qlearning)

    return plot_values


def plot_results(env_name, algos, n_runs=50, error="", ylabel="", limit_q_value=False, c_lr=1.0):
    # all colors: tab:blue, tab:orange, tab:green, tab:red, tab:purple
    #             tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # all line styles: -, --, -., :, ., +, X, o, v, *
    # linestyles = ["-", "--", "-", "-."]
    linestyles = ["-", "-", "--", "-."]

    # plot settings
    set_color = False
    set_linestyle = True
    set_ci = True  # set True: first step, second step, third step
    plot_log_value = False  # second step
    set_yscale = False  # third step
    set_semilog = True  # fourth step
    use_kappaenv = True  # set True if equivalence and sigma are computed using kappaenv not riverswim

    if error == "onenorm":
        plot_log_value = False
    elif error == "gap":
        plot_log_value = False
        set_yscale = False
        set_semilog = False
    elif error == "nstates_optimal":
        plot_log_value = False
    elif error == "nstates_suboptimal":
        plot_log_value = False
        set_yscale = False
        set_semilog = False

    if set_semilog:
        set_ci = False

    # generate figure save name
    # fig_name = "RiverswimEnv6_QL"
    flag_ql = False
    flag_ucb = False
    for algo in algos:
        if algo in ["qlearning", "qlearning_es"]:
            flag_ql = True
        elif algo in ["qlearning_ucb", "qlearning_es_ucb"]:
            flag_ucb = True

    if (flag_ql == True) and (flag_ucb == False):
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        linestyles = ["-", "--", "-", "-."]
    elif (flag_ql == False) and (flag_ucb == True):
        colors = ["tab:green", "tab:red", "tab:blue", "tab:orange"]
        linestyles = ["-", "-.", "-", "--"]

    T, ylims, plt_title = plot_title_ylims(env_name,
                                           error,
                                           use_kappaenv,
                                           plot_log_value,
                                           set_semilog,
                                           set_yscale,
                                           flag_ql,
                                           flag_ucb,
                                           limit_q_value,
                                           c_lr)

    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    if set_yscale:
        # plt.yscale('log') is the same as plt.semilogy
        plt.yscale('log')

    for i_algo, algo in enumerate(algos):
        print("Start plotting", algo)

        if algo == "qlearning":
            legend = "QL"
        elif algo == "qlearning_es":
            legend = "QL-ES"
        elif algo == "qlearning_ucb":
            legend = "UCBQ"
        elif algo == "qlearning_es_ucb":
            legend = "UCBQ-ES"

        plot_values = extract_data(n_runs, env_name, algo, error, plot_log_value, q_optimal, policy_optimal)

        plot_mean_CI(plot_values,
                     legend,
                     T,
                     n_samping=5,
                     set_ci=set_ci,
                     set_semilog=set_semilog,
                     line_width=2,
                     line_style=linestyles[i_algo],
                     color=colors[i_algo])

    plt.ylim(ylims[0], ylims[1])
    plt.xlim(0, T)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel('Time steps', fontsize=14)
    plt.title(plt_title, fontsize=15)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True)

    # plt.show()

    fig_save_directory = "./results/plots/{}/".format(env_name)
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
    fig_name = env_name + draw_type + note
    plt.savefig(fig_save_directory + fig_name + ".pdf")

    print("Finish plotting.")


if __name__ == '__main__':
    # envs: riverswim, riverswimkappa1, riverswimkappa2, riverswimkappa3
    #       gridworld, gridworldkappa1
    env_type = "gridworld"

    if env_type == 'riverswim':
        nbstates = 6
        env = RiverswimEnv(nS=nbstates)
        # env_name = env_type + str(nbstates)
        env_name = env_type + str(nbstates) + "_11"
    elif env_type == 'riverswimkappa1':
        nbstates = 20
        env = RiverswimEnvKappa1(nS=nbstates)
        # env_name = "riverswim{}kappa1".format(nbstates)
        env_name = "riverswim{}kappa1".format(nbstates) + "_6"
    elif env_type == 'riverswimkappa2':
        nbstates = 20
        env = RiverswimEnvKappa2(nS=nbstates)
        env_name = "riverswim{}kappa2".format(nbstates)
    elif env_type == 'riverswimkappa3':
        nbstates = 20
        env = RiverswimEnvKappa3(nS=nbstates)
        env_name = "riverswim{}kappa3".format(nbstates)
    elif env_type == 'gridworld':
        sizeX, sizeY = 11, 11
        # room name: 2room or 4room
        room_name = '4room'
        env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
        # env_name = env_type + room_name + "_" + str(sizeX) + "_" + str(sizeY)
        env_name = env_type + room_name + "_" + str(sizeX) + "_" + str(sizeY) + "_4"
    elif env_type == 'gridworldkappa1':
        sizeX, sizeY = 7, 7
        # room name: 2room or 4room
        room_name = '2room'
        env = GridworldEnv(sizeX=sizeX, sizeY=sizeY, room_name=room_name)
        # env_name = "gridworld{}_{}_{}_{}".format(room_name, sizeX, sizeY, "kappa1")
        env_name = "gridworld{}_{}_{}_{}_{}".format(room_name, sizeX, sizeY, "kappa1", 5)

    # algos = ["qlearning", "qlearning_es", "qlearning_ucb", "qlearning_es_ucb"]
    # algos = ["qlearning", "qlearning_ucb", "qlearning_es", "qlearning_es_ucb"]
    algos = ["qlearning_ucb", "qlearning_es_ucb"]
    # algos = ["qlearning", "qlearning_es"]
    # algos = ["qlearning_es"]
    # algos = ["qlearning"]
    # algos = ["qlearning_ucb"]

    # compute optimal q values and policy
    gamma = 0.85
    q_optimal, policy_optimal = q_value_iteration(env)
    # print(q_optimal)
    # print(policy_optimal)

    n_runs = 100
    limit_q_value = False  # set True if limit q value to 1/(1-\gamma) at each time step
    c_lr = 10.0  # learning rate coefficient
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
               "Optimal nstates",
               'Suboptimality nstates']
    i_error = 5
    print("env:", env_name)
    print("error type:", plot_errors[i_error])
    plot_results(env_name=env_name,
                 algos=algos,
                 n_runs=n_runs,
                 error=plot_errors[i_error],
                 ylabel=ylabels[i_error],
                 limit_q_value=limit_q_value,
                 c_lr=c_lr)
