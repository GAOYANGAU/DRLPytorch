# coding=utf-8
import random
#背包问题
#物品质量价格
X = {
    1: [10, 15],
    2: [15, 25],
    3: [20, 35],
    4: [25, 45],
    5: [30, 55],
    6: [35, 70]}

#终止界限
FINISHED_LIMIT = 5

#重量界限
WEIGHT_LIMIT = 80

#染色体长度
CHROMOSOME_SIZE = 6

#遴选次数
SELECT_NUMBER = 4

max_last = 0
diff_last = 10000

#判断退出
def is_finished(fitnesses):
    global max_last
    global diff_last

    max_current = 0
    for v in fitnesses:
        if v[1] > max_current:
            max_current = v[1]

    diff = max_current - max_last
    if diff < FINISHED_LIMIT and diff_last < FINISHED_LIMIT:
        return True
    else:
        diff_last = diff
        max_last = max_current
        return False

#初始染色体样态
def init():
    chromosome_state1 = '100100'
    chromosome_state2 = '101010'
    chromosome_state3 = '010101'
    chromosome_state4 = '101011'
    chromosome_states = [chromosome_state1,
                         chromosome_state2,
                         chromosome_state3,
                         chromosome_state4]
    return chromosome_states


#计算适应度
def fitness(chromosome_states):
    fitnesses = []
    for chromosome_state in chromosome_states:
        value_sum = 0
        weight_sum = 0
        for i, v in enumerate(chromosome_state):
            if int(v) == 1:
                weight_sum += X[i + 1][0]
                value_sum += X[i + 1][1]
        fitnesses.append([value_sum, weight_sum])
    return fitnesses


#筛选
def filter(chromosome_states, fitnesses):
    #重量大于80的被淘汰
    index = len(fitnesses) - 1
    while index >= 0:
        index -= 1
        if fitnesses[index][1] > WEIGHT_LIMIT:
            chromosome_states.pop(index)
            fitnesses.pop(index)

    #遴选
    selected_index = [0] * len(chromosome_states)
    for i in range(SELECT_NUMBER):
        j = chromosome_states.index(random.choice(chromosome_states))
        selected_index[j] += 1
    return selected_index


#产生下一代
def crossover(chromosome_states, selected_index):
    chromosome_states_new = []
    index = len(chromosome_states) - 1
    while index >= 0:
        index -= 1
        chromosome_state = chromosome_states.pop(index)
        for i in range(selected_index[index]):
            chromosome_state_x = random.choice(chromosome_states)
            pos = random.choice(range(1, CHROMOSOME_SIZE - 1))
            chromosome_states_new.append(chromosome_state[:pos] + chromosome_state_x[pos:])
        chromosome_states.insert(index, chromosome_state)
    return chromosome_states_new


if __name__ == '__main__':
    #初始群体
    chromosome_states = init()
    n = 100
    while n > 0:
        n -= 1
        #适应度计算
        fitnesses = fitness(chromosome_states)
        if is_finished(fitnesses):
            break
        print('1:', fitnesses)
        #遴选
        selected_index = filter(chromosome_states, fitnesses)
        print('2:', selected_index)
        #产生下一代
        chromosome_states = crossover(chromosome_states, selected_index)
        # print '3:', chromosome_states

    fitnesses = fitness(chromosome_states)
    print(chromosome_states)

# 1: [[60, 35], [105, 60], [140, 75], [175, 95]]
# 2: [1, 1, 2]

# 1: [[60, 35], [105, 60], [80, 45], [90, 50]]
# 2: [2, 1, 0, 1]

# 1: [[95, 55], [115, 65], [70, 40], [90, 50]]
# 2: [2, 0, 2, 0]

# 1: [[70, 40], [70, 40], [150, 85], [115, 65]]
# 2: [3, 0, 1]

# 1: [[115, 65], [115, 65], [115, 65], [70, 40]]
# 2: [2, 0, 0, 2]
# ['100110', '100110', '100110', '100110']
