import copy
import numpy as np

from utils import make_epsilon_greedy_policy


def sarsa(env, n_steps=300000, gamma=0.85, epsilon=0.1, n_steps_store=10, use_eps_decay=False,
          eps_decay=0.001):
    # init q
    Q_all = {}

    Q = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    count_s_a = {}
    s = env.reset()
    a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]

    for i_step in range(n_steps):

        # decay the epsilon value until it reaches the threshold of 0.01
        # epsilon decay better than no
        if use_eps_decay:
            if epsilon > 0.01:
                epsilon -= eps_decay

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        s_prime, r, _, _ = env.step(a)

        # count occurence of (s,a)
        if (s, a) in count_s_a:
            count_s_a[(s, a)] += 1
        else:
            count_s_a[(s, a)] = 1

        # update learning rate
        alpha = 10 / (count_s_a[(s, a)] + 1)

        a_prime = np.random.choice(range(env.action_space.n), 1, p=policy(s_prime))[0]
        Q[s][a] += alpha * (r + gamma * Q[s_prime][a_prime] - Q[s][a])

        s, a = s_prime, a_prime

        # store q values every n steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def sarsa_lambda(env,
                 n_steps=300000,
                 gamma=0.85,
                 epsilon=0.1,
                 n_steps_store=10,
                 use_eps_decay=False,
                 eps_decay=0.001,
                 trace_decay=0.9,
                 type='accumulate'):
    # the code is changed from: https://naifmehanna.com/2018-10-18-implementing-sarsa-in-python/

    # init q
    Q_all = {}

    Q = {}
    E = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA
        E[s] = [0] * env.nA

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    count_s_a = {}
    for s in range(env.nS):
        for a in range(env.nA):
            count_s_a[(s, a)] = 0

    s = env.reset()
    a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]

    for i_step in range(n_steps):

        # decay the epsilon value until it reaches the threshold of 0.01
        # epsilon decay better than no
        if use_eps_decay:
            if epsilon > 0.01:
                epsilon -= eps_decay

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        s_prime, r, _, _ = env.step(a)

        # count occurence of (s,a)
        count_s_a[(s, a)] += 1

        # first way: worse performance
        # alpha = 0.01

        # second way: worse than third way
        # alpha = 10 / (count_s_a[(s, a)] + 1)

        a_prime = np.random.choice(range(env.action_space.n), 1, p=policy(s_prime))[0]
        delta = r + gamma * Q[s_prime][a_prime] - Q[s][a]

        E[s][a] += 1

        for s_ in range(env.nS):
            for a_ in range(env.nA):
                # third way
                alpha = 10 / (count_s_a[(s_, a_)] + 1)

                Q[s_][a_] += alpha * delta * E[s_][a_]
                if type == 'accumulate':
                    E[s_][a_] *= trace_decay * gamma
                elif type == 'replace':
                    if s_ == s:
                        E[s_][a_] = 1
                    else:
                        E[s_][a_] *= gamma * trace_decay

        s, a = s_prime, a_prime

        # store q values every n steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all
