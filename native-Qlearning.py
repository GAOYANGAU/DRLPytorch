# 问题http://mnemstudio.org/path-finding-q-learning-tutorial.htm的Q学习方法实现
import numpy as np
import random
import matplotlib.pyplot as plt

Q_fun = np.zeros((6, 6))

# 回报函数，在状态state采用action转移到next_state的回报，横纵坐标分别为state和next_state
reward = np.array([[-1, -1, -1, -1, 0, -1],
              [-1, -1, -1, 0, -1, 100],
              [-1, -1, -1, 0, -1, -1],
              [-1, 0, 0, -1, 0, -1],
              [0, -1, -1, 0, -1, 100],
              [-1, 0, -1, -1, 0, 100]])

legal_action = [[4],
                [3, 5],
                [3],
                [1, 2, 4],
                [0, 3, 5],
                [1, 4, 5]]

GAMMA = 0.5
TRAINING_STEP = 100
LAYOUT = 221

for i in range(1, TRAINING_STEP + 1):
    state = random.randint(0, 4)
    # 百分百探索，随机产生next_state
    next_state = random.choice(legal_action[state])
    Q_fun[state, next_state] = reward[state, next_state] + GAMMA * Q_fun[next_state].max()

    if i % (TRAINING_STEP/4) == 0:
        plt.subplot(LAYOUT)
        plt.imshow(Q_fun, cmap='gray_r')
        LAYOUT += 1
        print(Q_fun)
plt.show()


