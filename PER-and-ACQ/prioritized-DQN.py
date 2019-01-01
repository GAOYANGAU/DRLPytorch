import torch
import torch.nn as nn
from prioritized_memory import Memory
import numpy as np
import gym
import random
from net import AtariNet
from util import preprocess

BATCH_SIZE = 32
LR = 0.001
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
TOTAL_EPISODES = 10000000
MEMORY_SIZE = 1000000
MEMORY_THRESHOLD = 100000
UPDATE_TIME = 10000
TEST_FREQUENCY = 1000
env = gym.make('Pong-v0')
env = env.unwrapped
ACTIONS_SIZE = env.action_space.n


class Agent(object):
    def __init__(self):
        self.network, self.target_network = AtariNet(ACTIONS_SIZE), AtariNet(ACTIONS_SIZE)
        self.memory = Memory(MEMORY_SIZE)
        self.learning_count = 0
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def action(self, state, israndom):
        if israndom and random.random() < EPSILON:
            return np.random.randint(0, ACTIONS_SIZE)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.network.forward(state)
        return torch.max(actions_value, 1)[1].data.numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        old_val = self.network.forward(torch.FloatTensor([state])).gather(1, torch.LongTensor([[action]]))[0]
        target_val = self.network.forward(torch.FloatTensor([state]))
        if done:
            done = 0
            target = reward
        else:
            done = 1
            target = reward + GAMMA * torch.max(target_val)
        error = abs(old_val[0] - target)
        self.memory.add(error.data, (state, action, reward, next_state, done))
        if self.memory.tree.n_entries < MEMORY_THRESHOLD:
            return

        if self.learning_count % UPDATE_TIME == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.learning_count += 1

        batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])

        eval_q = self.network.forward(state).gather(1, action)
        next_q = self.target_network(next_state).detach()
        target_q = reward + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done
        errors = torch.abs(eval_q - target_q).data.numpy().flatten()
        loss = self.loss_func(eval_q, target_q)

        for i in range(BATCH_SIZE):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


agent = Agent()

for i_episode in range(TOTAL_EPISODES):
    state = env.reset()
    state = preprocess(state)
    while True:
        # env.render()
        action = agent.action(state, True)
        next_state, reward, done, info = env.step(action)
        next_state = preprocess(next_state)
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        if done:
            break
    if EPSILON > FINAL_EPSILON:
        EPSILON -= (START_EPSILON - FINAL_EPSILON) / EXPLORE

    # TEST
    if i_episode % TEST_FREQUENCY == 0:
        state = env.reset()
        state = preprocess(state)
        total_reward = 0
        while True:
            # env.render()
            action = agent.action(state, israndom=False)
            next_state, reward, done, info = env.step(action)
            next_state = preprocess(next_state)

            total_reward += reward

            state = next_state
            if done:
                break
        print('episode: {} , total_reward: {}'.format(i_episode, round(total_reward, 3)))

env.close()
