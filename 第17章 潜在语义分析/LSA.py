"""
    Implementation of the Latent Semantic Analysis (LSA)
    written by zjxi @ 2020/12/05
"""

import numpy as np
import pandas as pd
import jieba


df = pd.read_csv("../data/train.txt")
data = np.array(df['abstract'])
dic = []
for i in data:
    cutdata = jieba.cut(i)
    for j in cutdata:
        if j not in dic and len(j)>=2:
            dic.append(j)
a = len(dic)
X = np.zeros((a, len(data)))

for i in range(len(data)):
    cutdata = jieba.cut(data[i])
    p = []
    for j in cutdata:
        p.append(j)
    for k in range(len(p)):
        if p[k] in dic:
            index = dic.index(p[k])
            X[index][i] += 1

U, O, V = np.linalg.svd(X)

print(U[:, :3])
a = np.diag(O[:3])
b = V[:3, :]
c = np.dot(a, b)
print(c.shape)