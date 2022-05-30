import numpy as np
import copy
from utils import make_epsilon_greedy_policy


def qlearning(env,
              n_steps=300000,
              gamma=0.85,
              epsilon=0.1,
              n_steps_store=10,
              c_lr=10.0,
              limit_q_value=False):
    # theorem q max: R_max / (1 - gamma), currently R_max=1
    theo_q_max = 1 / (1 - gamma)

    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA

    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    count_s_a = {}
    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count (s,a)
        if (s, a) in count_s_a:
            count_s_a[(s, a)] += 1
        else:
            count_s_a[(s, a)] = 1

        # update learning rate
        if c_lr < 1.0:
            alpha_t = c_lr
        else:
            alpha_t = c_lr / (count_s_a[(s, a)] + 1)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + gamma * max_q - Q[s][a])
        if limit_q_value:
            Q[s][a] = min(Q[s][a], theo_q_max)

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_ucb(
        env, n_steps=300000, gamma=0.85, epsilon=0.1, ucb_c=0.55, n_steps_store=10
):
    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    Q_hat = {}
    for s in range(env.nS):
        Q[s] = [1 / (1 - gamma)] * env.nA
        Q_hat[s] = [1 / (1 - gamma)] * env.nA

    # policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

    H = 1 / (1 - gamma)
    count_s_a = {}
    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count occurence of (s,a)
        if (s, a) in count_s_a:
            count_s_a[(s, a)] += 1
        else:
            count_s_a[(s, a)] = 1
        k = count_s_a[(s, a)]
        b_k = ucb_c * np.sqrt(H / k)
        alpha_t = (H + 1) / (H + k)

        max_q = np.max(Q_hat[s_prime])
        Q[s][a] += alpha_t * (r + b_k + gamma * max_q - Q[s][a])
        Q_hat[s][a] = np.min([Q[s][a], Q_hat[s][a]])

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_es(
        env,
        sigma_s_a,
        C_s_a,
        n_steps=300000,
        gamma=0.85,
        epsilon=0.1,
        n_steps_store=10,
        c_lr=10.0,
        limit_q_value=False,
        use_class_alpha=True
):
    theo_q_max = 1 / (1 - gamma)

    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA

    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # init es count
    if use_class_alpha:
        count_C_s_a = {}
        for c in range(len(C_s_a)):
            count_C_s_a[c] = 0
    else:
        count_s_a = {}
        for s in range(env.nS):
            for a in range(env.nA):
                count_s_a[(s, a)] = 0

    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count es and update learning rate
        if c_lr < 1.0:
            alpha_t = c_lr
        else:
            if use_class_alpha:
                for c in range(len(C_s_a)):
                    if (s, a) in C_s_a[c]:
                        count_C_s_a[c] += 1
                        alpha_t = c_lr / (count_C_s_a[c] + 1)
            else:
                count_s_a[(s, a)] += 1
                alpha_t = c_lr / (count_s_a[(s, a)] + 1)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + gamma * max_q - Q[s][a])
        if limit_q_value:
            Q[s][a] = min(Q[s][a], theo_q_max)

        for pairs_es in sigma_s_a[s, a, s_prime]:
            s_es, a_es = pairs_es[0]
            s_prime_es = pairs_es[1]
            r_es = pairs_es[2]
            max_q_es = np.max(Q[s_prime_es])
            if c_lr < 1.0:
                alpha_t = c_lr
            else:
                if not use_class_alpha:
                    alpha_t = c_lr / (count_s_a[(s_es, a_es)] + 1)
            Q[s_es][a_es] += alpha_t * (r_es + gamma * max_q_es - Q[s_es][a_es])
            if limit_q_value:
                Q[s_es][a_es] = min(Q[s_es][a_es], theo_q_max)

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    # print(alpha_t)

    return Q_all


def qlearning_es_ucb(
        env,
        sigma_s_a,
        C_s_a,
        n_steps=300000,
        gamma=0.85,
        epsilon=0.1,
        ucb_c=0.55,
        n_steps_store=10,
):
    # mix using c (es and ucb c) in yijie's code
    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    Q_hat = {}
    for s in range(env.nS):
        Q[s] = [1 / (1 - gamma)] * env.nA
        Q_hat[s] = [1 / (1 - gamma)] * env.nA

    # policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

    # init es count
    count_C_s_a = {}
    for c in range(len(C_s_a)):
        count_C_s_a[c] = 0

    H = 1 / (1 - gamma)
    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count es and update learning rate
        for c in range(len(C_s_a)):
            if (s, a) in C_s_a[c]:
                count_C_s_a[c] += 1
                k = count_C_s_a[c]
                b_k = ucb_c * np.sqrt(H / k)
                alpha_t = (H + 1) / (H + k)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + b_k + gamma * max_q - Q[s][a])
        Q_hat[s][a] = np.min([Q[s][a], Q_hat[s][a]])

        for pairs_es in sigma_s_a[s, a, s_prime]:
            s_es, a_es = pairs_es[0]
            s_prime_es = pairs_es[1]
            r_es = pairs_es[2]
            max_q_es = np.max(Q[s_prime_es])
            Q[s_es][a_es] += alpha_t * (r_es + b_k + gamma * max_q_es - Q[s_es][a_es])
            Q_hat[s_es][a_es] = np.min([Q[s_es][a_es], Q_hat[s_es][a_es]])

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_es_few(env,
                     sigma_s_a,
                     C_s_a,
                     n_steps=300000,
                     gamma=0.85,
                     epsilon=0.1,
                     n_steps_store=10,
                     n_es_pairs=1,
                     c_lr=10.0,
                     limit_q_value=False):
    theo_q_max = 1 / (1 - gamma)

    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA

    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # init es count
    count_C_s_a = {}
    for c in range(len(C_s_a)):
        count_C_s_a[c] = 0

    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count es and update learning rate
        if c_lr < 1.0:
            alpha_t = c_lr
        else:
            for c in range(len(C_s_a)):
                if (s, a) in C_s_a[c]:
                    count_C_s_a[c] += 1
                    alpha_t = c_lr / (count_C_s_a[c] + 1)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + gamma * max_q - Q[s][a])
        if limit_q_value:
            Q[s][a] = min(Q[s][a], theo_q_max)

        # randomly choose n pairs to update each time
        all_pairs = sigma_s_a[s, a, s_prime]
        n_c = len(all_pairs)
        if n_c <= n_es_pairs:
            es_few_pairs = all_pairs
        else:
            index_all_pairs = range(n_c)
            index_es = sorted(np.random.choice(index_all_pairs, n_es_pairs, replace=False))
            es_few_pairs = []
            for i_index in index_es:
                es_few_pairs.append(all_pairs[i_index])

        for pairs_es in es_few_pairs:
            s_es, a_es = pairs_es[0]
            s_prime_es = pairs_es[1]
            r_es = pairs_es[2]
            max_q_es = np.max(Q[s_prime_es])
            Q[s_es][a_es] += alpha_t * (r_es + gamma * max_q_es - Q[s_es][a_es])
            if limit_q_value:
                Q[s_es][a_es] = min(Q[s_es][a_es], theo_q_max)

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_es_ucb_few(
        env,
        sigma_s_a,
        C_s_a,
        n_steps=300000,
        gamma=0.85,
        epsilon=0.1,
        ucb_c=0.55,
        n_steps_store=10,
        n_es_pairs=1
):
    # mix using c (es and ucb c) in yijie's code
    # a dict to store q every n_steps_store
    Q_all = {}

    # init q
    Q = {}
    Q_hat = {}
    for s in range(env.nS):
        Q[s] = [1 / (1 - gamma)] * env.nA
        Q_hat[s] = [1 / (1 - gamma)] * env.nA

    # policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

    # init es count
    count_C_s_a = {}
    for c in range(len(C_s_a)):
        count_C_s_a[c] = 0

    H = 1 / (1 - gamma)
    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q_hat, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count es and update learning rate
        for c in range(len(C_s_a)):
            if (s, a) in C_s_a[c]:
                count_C_s_a[c] += 1
                k = count_C_s_a[c]
                b_k = ucb_c * np.sqrt(H / k)
                alpha_t = (H + 1) / (H + k)

        max_q = np.max(Q[s_prime])
        Q[s][a] += alpha_t * (r + b_k + gamma * max_q - Q[s][a])
        Q_hat[s][a] = np.min([Q[s][a], Q_hat[s][a]])

        # randomly choose n pairs to update each time
        all_pairs = sigma_s_a[s, a, s_prime]
        n_c = len(all_pairs)
        if n_c <= n_es_pairs:
            es_few_pairs = all_pairs
        else:
            index_all_pairs = range(n_c)
            index_es = sorted(np.random.choice(index_all_pairs, n_es_pairs, replace=False))
            es_few_pairs = []
            for i_index in index_es:
                es_few_pairs.append(all_pairs[i_index])

        for pairs_es in es_few_pairs:
            s_es, a_es = pairs_es[0]
            s_prime_es = pairs_es[1]
            r_es = pairs_es[2]
            max_q_es = np.max(Q[s_prime_es])
            Q[s_es][a_es] += alpha_t * (r_es + b_k + gamma * max_q_es - Q[s_es][a_es])
            Q_hat[s_es][a_es] = np.min([Q[s_es][a_es], Q_hat[s_es][a_es]])

        s = s_prime

        # store q values every n_steps_store steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all


def qlearning_lambda(env,
                     n_steps=300000,
                     gamma=0.85,
                     epsilon=0.1,
                     n_steps_store=10,
                     trace_decay=0.9,
                     c_lr=10.0,
                     type='accumulate'):
    # similar code: https://github.com/vinhvu200/Windy-Grid-World/blob/master/Q-Learning.ipynb
    # init q
    Q_all = {}

    Q = {}
    E = {}
    for s in range(env.nS):
        Q[s] = [0] * env.nA
        E[s] = [0] * env.nA

    # policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    count_s_a = {}
    for s in range(env.nS):
        for a in range(env.nA):
            count_s_a[(s, a)] = 0

    s = env.reset()
    for i_step in range(n_steps):

        # update policy
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

        a = np.random.choice(range(env.action_space.n), 1, p=policy(s))[0]
        s_prime, r, _, _ = env.step(a)

        # count occurence of (s,a)
        count_s_a[(s, a)] += 1

        max_q = np.max(Q[s_prime])
        delta = r + gamma * max_q - Q[s][a]

        E[s][a] += 1

        for s_ in range(env.nS):
            for a_ in range(env.nA):
                if c_lr < 1.0:
                    alpha = c_lr
                else:
                    alpha = c_lr / (count_s_a[(s_, a_)] + 1)

                Q[s_][a_] += alpha * delta * E[s_][a_]
                if type == 'accumulate':
                    E[s_][a_] *= trace_decay * gamma
                elif type == 'replace':
                    if s_ == s:
                        E[s_][a_] = 1
                    else:
                        E[s_][a_] *= gamma * trace_decay

        s = s_prime

        # store q values every n steps
        if (i_step + 1) % n_steps_store == 0:
            Q_all[i_step] = copy.deepcopy(Q)

    return Q_all
