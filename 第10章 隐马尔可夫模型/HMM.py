"""
    Implementation of the Hidden Markov Model (HMM),
    written by zjxi @ 2020/11/17
"""

import numpy as np

# 状态 1 2 3
A = [[0.5, 0.2, 0.3],
     [0.3, 0.5, 0.2],
     [0.2, 0.3, 0.5]]

pi = [0.2, 0.4, 0.4]

# red white
B = [[0.5, 0.5],
     [0.4, 0.6],
     [0.7, 0.3]]


# 前向算法
def hmm_forward(A, B, pi, O):
    T = len(O)
    N = len(A[0])
    # step1 初始化
    alpha = [[0] * T for _ in range(N)]
    for i in range(N):
        alpha[i][0] = pi[i] * B[i][O[0]]

    # step2 计算alpha(t)
    for t in range(1, T):
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += alpha[j][t - 1] * A[j][i]
            alpha[i][t] = temp * B[i][O[t]]

    # step3
    proba = 0
    for i in range(N):
        proba += alpha[i][-1]
    return proba, alpha


A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
pi = [0.2, 0.4, 0.4]
O = [0, 1, 0, 1]
print(hmm_forward(A, B, pi, O))  # acc为 0.06009


# 后向算法
def hmm_backward(A, B, pi, O):
    T = len(O)
    N = len(A[0])
    # step1 初始化
    beta = [[0] * T for _ in range(N)]
    for i in range(N):
        beta[i][-1] = 1

    # step2 计算beta(t)
    for t in reversed(range(T - 1)):
        for i in range(N):
            for j in range(N):
                beta[i][t] += A[i][j] * B[j][O[t + 1]] * beta[j][t + 1]

    # step3
    proba = 0
    for i in range(N):
        proba += pi[i] * B[i][O[0]] * beta[i][0]
    return proba, beta


A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
pi = [0.2, 0.4, 0.4]
O = [0, 1, 0, 1]
print(hmm_backward(A, B, pi, O))  # acc为 0.06009


# 维特比算法
def hmm_viterbi(A, B, pi, O):
    T = len(O)
    N = len(A[0])

    delta = [[0] * N for _ in range(T)]
    psi = [[0] * N for _ in range(T)]

    # step1: init
    for i in range(N):
        delta[0][i] = pi[i] * B[i][O[0]]
        psi[0][i] = 0

    # step2: iter
    for t in range(1, T):
        for i in range(N):
            temp, maxindex = 0, 0
            for j in range(N):
                res = delta[t - 1][j] * A[j][i]
                if res > temp:
                    temp = res
                    maxindex = j

            delta[t][i] = temp * B[i][O[t]]  # delta
            psi[t][i] = maxindex

    # step3: end
    p = max(delta[-1])
    for i in range(N):
        if delta[-1][i] == p:
            i_T = i

    # step4：backtrack
    path = [0] * T
    i_t = i_T
    for t in reversed(range(T - 1)):
        i_t = psi[t + 1][i_t]
        path[t] = i_t
    path[-1] = i_T

    return delta, psi, path


A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
pi = [0.2, 0.4, 0.4]
O = [0, 1, 0, 1]
print(hmm_viterbi(A, B, pi, O))


