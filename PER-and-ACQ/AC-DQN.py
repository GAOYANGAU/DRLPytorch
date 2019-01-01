from __future__ import print_function
import torch, time, gym, argparse, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from collections import deque
import random
from net import AtariNet
from util import preprocess

LR = 0.001
EXPLORE = 1000000
GAMMA = 0.99
N_STEP = 4
ENV = 'Pong-v0'
ACTIONS_SIZE = gym.make(ENV).action_space.n
PROCESSES = 1
SEED = 1


class Agent(object):
    def __init__(self, action_size):
        self.action_size = action_size
        self.EPSILON = 1.0
        self.network = AtariNet(action_size)
        self.memory = deque()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def action(self, state, israndom):
        if israndom and random.random() < self.EPSILON:
            return np.random.randint(0, self.action_size)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.network.forward(state)
        return torch.max(actions_value, 1)[1].data.numpy()[0]

    def add(self, state, action, reward, next_state, done):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))

    def learn(self, shared_optimizer, shared_model):
        batch_size = len(self.memory)
        batch = random.sample(self.memory, batch_size)
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])

        eval_q = self.network.forward(state).gather(1, action)
        next_q = self.network(next_state).detach()
        target_q = reward + GAMMA * next_q.max(1)[0].view(batch_size, 1) * done
        loss = self.loss_func(eval_q, target_q)

        shared_optimizer.zero_grad()
        loss.backward()
        for param, shared_param in zip(self.network.parameters(), shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        shared_optimizer.step()

        self.memory = deque()
        if self.EPSILON > 0.1:
            self.EPSILON -= (1.0 - 0.1) / EXPLORE


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()


def worker(shared_model, shared_optimizer, rank, info):
    env = gym.make(ENV)
    env.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    agent = Agent(ACTIONS_SIZE)

    start_time = last_disp_time = time.time()
    episode_length, epr = 0, 0

    state = env.reset()
    state = preprocess(state)
    while info['frames'][0] <= 4e7:
        agent.network.load_state_dict(shared_model.state_dict())

        for _ in range(N_STEP):
            # env.render()
            episode_length += 1

            action = agent.action(state, True)
            next_state, reward, done, ext = env.step(action)
            epr += reward
            done = done or episode_length >= 1e4
            info['frames'].add_(1)
            num_frames = int(info['frames'].item())

            next_state = preprocess(next_state)
            agent.add(state, action, reward, next_state, done)

            state = next_state

            if done:
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 0.01
                info['run_epr'].mul_(1 - interp).add_(interp * epr)

            if rank == 0 and time.time() - last_disp_time > 60:
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                print('time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}'
                         .format(elapsed, info['episodes'].item(), num_frames / 1e6,
                                 info['run_epr'].item()))
                last_disp_time = time.time()

            if done:
                episode_length, epr, eploss = 0, 0, 0
                state = env.reset()
                state = preprocess(state)
                break

        agent.learn(shared_optimizer, shared_model)



if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn')
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux! Or else you get a deadlock in conv2d"

    torch.manual_seed(SEED)
    shared_model = AtariNet(ACTIONS_SIZE).share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=LR)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'episodes', 'frames']}

    processes = []
    for rank in range(PROCESSES):
        p = mp.Process(target=worker, args=(shared_model, shared_optimizer, rank, info))
        p.start()
        processes.append(p)
    for p in processes: p.join()
