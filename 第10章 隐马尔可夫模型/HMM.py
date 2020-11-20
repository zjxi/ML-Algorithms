"""
    Implementation of the Hidden Markov Model (HMM),
    written by zjxi @ 2020/11/17
"""

import numpy as np


class HMM:
    def __init__(self):
        # 初始概率分布
        self.pi = [0.2, 0.4, 0.4]

        # 状态转移概率分布矩阵 (1、2、3)
        self.A = [[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]]

        # 观测概率分布(红、白)
        self.B = [[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]]

        # 观测序列 (0:红色, 1:白色)
        self.O = [0, 1, 0, 1]

    def hmm_forward(self):
        """
        观测序列概率的前向算法
        :return: 观测序列概率P(O|λ)、前向概率alpha
        """
        # 算法10.2的流程
        # 观测序列总数
        T = len(self.O)
        # 状态总数
        N = len(self.A[0])

        # (1) 初值
        alpha = np.zeros((N, T))
        for i in range(N):
            alpha[i][0] = self.pi[i] * self.B[i][self.O[0]]

        # (2) 递推, 计算前向概率alpha(t)
        for t in range(1, T):
            for i in range(N):
                temp = 0
                for j in range(N):
                    temp += alpha[j][t - 1] * self.A[j][i]
                alpha[i][t] = temp * self.B[i][self.O[t]]

        # (3) 终止
        prob = 0
        for i in range(N):
            prob += alpha[i][-1]

        return prob, alpha

    def hmm_backward(self):
        """
        观测序列概率的后向算法
        :return: 观测序列概率P(O|λ)、后向概率beta
        """
        # 算法10.3的流程
        T = len(self.O)
        N = len(self.A[0])

        # (1) 初值
        beta = np.zeros((N, T))
        for i in range(N):
            beta[i][-1] = 1

        # (2) 递推, 计算后向概率beta(t)
        for t in reversed(range(T - 1)):
            for i in range(N):
                for j in range(N):
                    beta[i][t] += self.A[i][j] * self.B[j][self.O[t + 1]] * beta[j][t + 1]

        # (3) 终止
        prob = 0
        for i in range(N):
            prob += self.pi[i] * self.B[i][self.O[0]] * beta[i][0]

        return prob, beta

    def hmm_viterbi(self):
        """
        维特比算法
        :return: 最优路径I*
        """
        # 算法10.5的流程
        global i_T
        T = len(self.O)
        N = len(self.A[0])

        delta = np.zeros((T, N))
        psi   = np.zeros((T, N), dtype=int)

        # (1) 初始化
        for i in range(N):
            delta[0][i] = self.pi[i] * self.B[i][self.O[0]]
            psi[0][i] = 0

        # (2) 递推
        for t in range(1, T):
            for i in range(N):
                temp, max_index = 0, 0
                for j in range(N):
                    res = delta[t - 1][j] * self.A[j][i]
                    if res > temp:
                        temp = res
                        max_index = j

                delta[t][i] = temp * self.B[i][self.O[t]]  # delta
                psi[t][i] = max_index

        # (3) 终止
        p = max(delta[-1])
        for i in range(N):
            if delta[-1][i] == p:
                i_T = i

        # (4) 最优路径回溯
        path = [0] * T
        i_t = i_T
        for t in reversed(range(T - 1)):
            i_t = psi[t + 1][i_t]
            path[t] = i_t
        path[-1] = i_T

        return path


if __name__ == '__main__':
    # HMM算法初始化
    hmm = HMM()

    # 前向算法
    print(hmm.hmm_forward())

    # 后向算法
    print(hmm.hmm_backward())

    # 维特比算法
    print(hmm.hmm_viterbi())
