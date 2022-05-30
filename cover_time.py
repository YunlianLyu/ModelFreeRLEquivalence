import numpy as np

from environments.riverswim import RiverswimEnv
from utils import equivalenceClasses, make_uniform_policy


def compute_t_cover_rs(env, n_steps=300000):
    n_all_s_a = env.nS * env.nA

    count_s_a = 0
    contain_s_a = []

    t_cover = []
    policy = make_uniform_policy(env.action_space.n)

    s = env.reset()
    for i_step in range(n_steps):

        a = np.random.choice(range(env.action_space.n), 1, p=policy())[0]
        s_prime, _, _, _ = env.step(a)

        if (s, a) not in contain_s_a:
            contain_s_a.append((s, a))
            count_s_a += 1
            if count_s_a == n_all_s_a:
                t_cover.append(i_step)
                contain_s_a = []
                count_s_a = 0

        s = s_prime

    return t_cover


def compute_t_cover_rs_es(env, n_steps=300000):
    n_all_s_a = 3

    count_s_a = 0
    contain_s_a = []

    t_cover = []
    policy = make_uniform_policy(env.action_space.n)

    s = env.reset()
    for i_step in range(n_steps):

        a = np.random.choice(range(env.action_space.n), 1, p=policy())[0]
        s_prime, _, _, _ = env.step(a)

        for c in range(n_all_s_a):
            if (s, a) in C_s_a_rs[c] and c not in contain_s_a:
                contain_s_a.append(c)
                count_s_a += 1
                if count_s_a == n_all_s_a:
                    t_cover.append(i_step)
                    contain_s_a = []
                    count_s_a = 0

        s = s_prime

    return t_cover


if __name__ == '__main__':
    n_runs = 100
    T = 300000

    env = RiverswimEnv(nS=2)

    C_s_a_rs, _ = equivalenceClasses(env=env, eps=0.0)
    print("*" * 100)
    print("equivalence structure:")
    for c in C_s_a_rs:
        print(c)
    print("*" * 100)

    test_1run = True
    if test_1run:
        t_cover = compute_t_cover_rs(env, T)
        print(t_cover)
        n1 = np.average(np.diff(t_cover))
        print(n1)

        t_cover_es = compute_t_cover_rs_es(env, T)
        print(t_cover_es)
        n2 = np.average(np.diff(t_cover_es))
        print(n2)
    else:
        t_cover_runs = []
        t_cover_es_runs = []
        for i_run in range(n_runs):
            print(i_run)
            t_cover = compute_t_cover_rs(env, T)
            print(t_cover)
            n1 = np.average(np.diff(t_cover))
            print(n1)

            t_cover_es = compute_t_cover_rs_es(env, T)
            print(t_cover_es)
            n2 = np.average(np.diff(t_cover_es))
            print(n2)

            t_cover_runs.append(t_cover)
            t_cover_es_runs.append(t_cover_es)

        t_cover_avg = np.average(t_cover_runs)
        t_cover_es_avg = np.average(t_cover_es_runs)

        print("riverswim{} cover time:".format(str(env.nS)), t_cover_avg)
        print("riverswim{} cover time es:".format(str(env.nS)), t_cover_es_avg)
