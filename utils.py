import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import gym


def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        idx_max = np.argmax(Q[observation])
        probabilities = np.ones(nA) * (epsilon / nA)
        probabilities[idx_max] += 1.0 - epsilon
        return probabilities

    return policy_fn


def make_uniform_policy(nA):
    def policy_fn():
        probabilities = np.ones(nA) * (1.0 / nA)
        return probabilities

    return policy_fn


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def get_transition(env, s, a):
    p = np.zeros(env.nS)
    for tran in env.P[s][a]:
        prob, next_state, reward, terminal = tran
        p[next_state] = prob
    return p


def compare(env, s1, a1, s2, a2):
    t1 = get_transition(env, s1, a1)
    t2 = get_transition(env, s2, a2)
    # r1 = env.get_reward(s1, a1)
    # r2 = env.get_reward(s2, a2)
    st1 = sorted(t1)
    st2 = sorted(t2)
    err = 0
    for i in range(env.nS):
        err += abs(st1[i] - st2[i])
    # err += abs(r1 - r2)
    return err


def equivalenceClass(env, s, a, eps):
    equiva = []
    for s2 in range(env.nS):
        for a2 in range(env.nA):
            if (compare(env, s, a, s2, a2) <= eps):
                # equiva.append([s2, a2])
                equiva.append((s2, a2))
    return equiva


# compute the equivalence structure
def equivalenceClasses(env, eps):
    eqClasses = []
    stateActionPairs = []
    saSize = 0
    nbEqClasses = 0
    indexEqClass = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            # stateActionPairs.append([s, a])
            stateActionPairs.append((s, a))
            saSize += 1
    while (saSize > 0):
        s, a = stateActionPairs.pop()
        saSize -= 1
        eqC = equivalenceClass(env, s, a, eps)
        eqClasses.append(eqC)
        nbEqClasses += 1
        indexEqClass[s][a] = nbEqClasses - 1
        for e in eqC:
            if e in stateActionPairs:
                s, a = e
                indexEqClass[s][a] = nbEqClasses - 1
                stateActionPairs.remove(e)
                saSize -= 1
    return eqClasses, indexEqClass


def compute_sigma(env):
    sigma = np.zeros((env.nS, env.nA, env.nS), dtype=int)
    for s in range(env.nS):
        for a in range(env.nA):
            li = list(np.argsort(get_transition(env, s, a)))
            sigma[s, a] = np.array(li)
    return sigma


# compute next states for equivalence classes according to the rank of probs
def profile_mapping(env, C):
    # sigma: show index according to probability
    sigma = compute_sigma(env)
    sigma_s_a = {}
    # sigma_s,a: store ((state, action), next_state, reward) for (state, action, next_state)
    for es_C in C:
        # es_C: equivalent classes, such as [(0, 1), (5, 1)]
        for es_s_a in es_C:
            # es_s_a: state-action pair in es_C, such as (0, 1)
            s, a = es_s_a
            # equivalence class for (s, a)
            es_classes = []
            for e_ in es_C:
                if e_ != (s, a):
                    es_classes.append(e_)
            # get all next states for (s,a)
            next_states = []
            for tran in env.P[s][a]:
                prob, s_prime, reward, terminal = tran
                next_states.append(s_prime)
            for next_state in next_states:
                sigma_s_a[(s, a, next_state)] = []
                # find ((state, action), next_state, reward) for (s, a, s')
                x = sigma[s, a]
                # probability rank index
                idx = np.argwhere(x == next_state).flatten()[0]
                # find next states of equivalent state-aciton pairs for (s, a)
                for class_pair in es_classes:
                    es_s, es_a = class_pair
                    es_prime = sigma[es_s, es_a, idx]
                    # env.P[s][a], such as [(0.6, 5, 1, 0), (0.4, 4, 1, 0)]
                    for tran in env.P[es_s][es_a]:
                        if tran[1] == es_prime:
                            es_r = tran[2]
                    sigma_s_a[(s, a, next_state)].append(((es_s, es_a), es_prime, es_r))
    return sigma_s_a


def q_value_iteration(env, theta=0.001, gamma=0.85, V_star=False):
    Q = np.zeros([env.nS, env.nA])

    while True:
        delta = 0
        Q_old = copy.deepcopy(Q)
        for state in env.P:
            for action in env.P[state]:
                sum_q = 0.0
                for transition_tuple in env.P[state][action]:
                    prob, next_state, reward, done = transition_tuple
                    sum_q += prob * (reward + gamma * np.max(Q_old[next_state]))
                Q[state][action] = sum_q
                delta = np.maximum(delta, np.abs(Q[state][action] - Q_old[state][action]))
        if delta < theta:
            break

    policy = [np.argmax(Q[i, :]) for i in range(env.nS)]

    if V_star:
        V = [max(Q[i, :]) for i in range(env.nS)]
        return Q, policy, V
    return Q, policy


def discretize(env, state):
    state = (state - env.observation_space.low) * np.array([10, 100])
    state = np.round(state, 0).astype(int)
    return state


def optimal_q_mcenv(gamma=0.85, theta=0.001, V_star=False):
    env = gym.make('MountainCar-v0')
    n_state = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
    n_state = np.round(n_state, 0).astype(int) + 1

    nS = n_state[0] * n_state[1]
    nA = 3

    mapping_s_to_pv = {}
    mapping_pv_to_s = {}
    i = 0
    for pos in range(n_state[0]):
        for vel in range(n_state[1]):
            mapping_s_to_pv[i] = (pos, vel)
            mapping_pv_to_s[(pos, vel)] = i
            i += 1

    Q = np.zeros([nS, nA])

    env.reset()
    while True:
        delta = 0
        Q_old = copy.deepcopy(Q)
        for state in range(nS):
            pos, vel = mapping_s_to_pv[state]
            env.state = [(pos / 10) - 1.2, (vel / 100) - 0.07]
            for action in range(nA):
                next_state, reward, done, info = env.step(action)
                next_state_apx = discretize(env, next_state)
                next_state = mapping_pv_to_s[(next_state_apx[0], next_state_apx[1])]
                Q[state][action] = (reward + gamma * np.max(Q_old[next_state]))
                delta = np.maximum(delta, np.abs(Q[state][action] - Q_old[state][action]))
        if delta < theta:
            break

    policy = [np.argmax(Q[i, :]) for i in range(nS)]

    if V_star:
        V = [max(Q[i, :]) for i in range(nS)]
        return Q, policy, V
    return Q, policy


def dict_max_v(Q):
    # Q table is saved every n steps
    # Q is a dict, like {{0}:[[],[]],...}
    vs = []
    for k in Q.keys():
        dq = Q[k]
        key, value = max(dq.items(), key=lambda x: x[1])
        vs.append(value)
    v_max = np.max(np.array(vs))
    t_v_max = np.argmax(np.array(vs))
    return v_max, t_v_max


def plot_mean_CI(results,
                 label,
                 n_steps,
                 n_samping=0,
                 set_ci=False,
                 set_semilog=False,
                 line_width=1,
                 line_style='-',
                 color=None,
                 marker=None,
                 marker_every=None):
    # results are saved every n steps
    results_mean = np.mean(results, axis=0)
    x = np.linspace(0.0, n_steps, num=results.shape[1])
    # plot data every n_sampling steps
    if n_samping > 0:
        results_mean = results_mean[0:n_steps:n_samping]
        x = x[0:n_steps:n_samping]
    if set_ci:
        # current problem: error brand is not symmetrical when using plt.yscale('log') or plt.semilogy
        results_sde = 1.96 * np.std(results, axis=0) / np.sqrt(results.shape[0])
        if n_samping > 0:
            results_sde = results_sde[0:n_steps:n_samping]
        plt.plot(
            x,
            results_mean,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color,
            marker=marker,
            markevery=marker_every
        )
        # color changes if set color=None using fill_between, so it is better to set exact color
        plt.fill_between(
            x,
            results_mean + results_sde,
            results_mean - results_sde,
            alpha=0.2,
            color=color
        )
    elif set_semilog:
        plt.semilogy(
            x,
            results_mean,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color,
            marker=marker,
            markevery=marker_every
        )
    else:
        plt.plot(
            x,
            results_mean,
            label=label,
            linewidth=line_width,
            linestyle=line_style,
            color=color,
            marker=marker,
            markevery=marker_every
        )
