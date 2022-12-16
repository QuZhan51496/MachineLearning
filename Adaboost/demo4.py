import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log


class Node:
    def __init__(self, label, value, axis, mode, err):
        self.label = label  # 最优属性
        self.axis = axis  # 最优属性列
        self.value = value  # 最优阈值
        self.mode = mode  # mode=0：低于阈值为女，高于阈值为男；mode=1：低于阈值为男，高于阈值为女
        self.err = err  # 最低权重错误率


train_set = pd.read_csv('traindata.csv')
train_set = train_set.values.tolist()
test_set = pd.read_csv('testdata.csv')
test_set = test_set.values.tolist()
label_name = ["身高", "体重"]
n = len(train_set)
m = len(test_set)


def get_err(data, axis, value, mode):
    w = data[:, -1]
    err = 0
    if mode == 0:  # 低于阈值为女，高于阈值为男
        for i in range(n):
            if (data[i][axis] <= value and data[i][-2] == 1) or (data[i][axis] > value and data[i][-2] == -1):
                err += w[i]
    else:  # 低于阈值为男，高于阈值为女
        for i in range(n):
            if (data[i][axis] <= value and data[i][-2] == -1) or (data[i][axis] > value and data[i][-2] == 1):
                err += w[i]
    return err


def build(data, name):
    step_num = 20
    min_err = 1
    for i in range(len(data[0]) - 2):
        min_values = np.min(data[:, i])
        max_values = np.max(data[:, i])
        step = (max_values - min_values) / step_num
        for j in range(step_num):
            now_value = min_values + j * step
            for k in range(2):
                err = get_err(data, i, now_value, k)
                if err < min_err:
                    min_err = err
                    best_axis = i
                    best_value = now_value
                    best_mode = k
    return Node(name[best_axis], best_value, best_axis, best_mode, min_err)


def update(data, rt):
    alpha = 0.5 * log((1 - rt.err) / rt.err)
    y = data[:, -2]
    h = np.zeros(n)
    if rt.mode == 0:  # 低于阈值为女，高于阈值为男
        for i in range(n):
            h[i] = -1 if data[i][rt.axis] <= rt.value else 1
    else:  # 低于阈值为男，高于阈值为女
        for i in range(n):
            h[i] = 1 if data[i][rt.axis] <= rt.value else -1
    tmp = data[:, -1] * np.exp(-alpha * y * h)
    data[:, -1] = tmp / sum(tmp)
    return alpha, h


def out(t, ac, rt):
    print("T = %d:" % t)
    print("\tAccuracy = %.3f" % ac)
    if rt.mode == 0:
        print("\t%s <= %.1f: 女" % (rt.label, rt.value))
        print("\t%s > %.1f: 男" % (rt.label, rt.value))
    else:
        print("\t%s <= %.1f: 男" % (rt.label, rt.value))
        print("\t%s > %.1f: 女" % (rt.label, rt.value))


def draw(t, rt):
    plt.subplot(3, 4, t)
    plt.title("T=%d" % t)
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.scatter(male[:, 0], male[:, 1], marker="o", color="b", label="Male")
    plt.scatter(female[:, 0], female[:, 1],
                marker="o", color="r", label="Female")
    for i in range(t):
        if rt[i].axis == 0:
            plt.plot(np.array([rt[i].value, rt[i].value]), np.array([40, 70]))
        else:
            plt.plot(np.array([155, 180]), np.array(
                [rt[i].value, rt[i].value]))
    plt.legend()


# 标准化标签，初始化权重
w = np.array([1 / n for t in range(n)])
for i in range(n):
    train_set[i][-1] = 1 if train_set[i][-1] == "男" else -1
for i in range(m):
    test_set[i][-1] = 1 if test_set[i][-1] == "男" else -1
train_set = np.array(train_set)
test_set = np.array(test_set)
train_set = np.hstack((train_set, w.reshape(len(w), 1)))

# 绘图预处理
male = []
female = []
for i in range(n):
    if train_set[i, -2] == 1:
        male.append([train_set[i][0], train_set[i][1]])
    else:
        female.append([train_set[i][0], train_set[i][1]])
male = np.array(male)
female = np.array(female)

# Adaboost算法
root = []
T = 12
alpha = np.zeros(T)
h = np.zeros((T, n))
plt.figure(figsize=(16, 12), dpi=90)
for t in range(T):
    root.append(build(train_set, label_name))
    alpha[t], h[t] = update(train_set, root[-1])
    ans = alpha.T @ h
    accuracy = 0
    for i in range(n):
        ans[i] = 1 if ans[i] > 0 else -1
        accuracy = accuracy + 1 if ans[i] == train_set[i, -2] else accuracy
    accuracy = accuracy / n
    out(t + 1, accuracy, root[-1])  # 输出准确率、弱分类器
    draw(t + 1, root)  # 绘图
plt.tight_layout()
plt.savefig('Adaboost.png')

# 测试
h_test = np.zeros((T, m))
for t in range(T):
    now = root[t]
    if now.mode == 0:  # 低于阈值为女，高于阈值为男
        for i in range(m):
            h_test[t][i] = -1 if test_set[i][now.axis] <= now.value else 1
    else:  # 低于阈值为男，高于阈值为女
        for i in range(m):
            h_test[t][i] = 1 if test_set[i][now.axis] <= now.value else -1
ans_test = alpha.T @ h_test
accuracy_test = 0
for i in range(m):
    ans_test[i] = 1 if ans_test[i] > 0 else -1
    accuracy_test = accuracy_test + \
        1 if ans_test[i] == test_set[i, -1] else accuracy_test
accuracy_test = accuracy_test / m
print('测试集准确率：', accuracy_test)
