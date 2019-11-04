import neat
import sys
import numpy as np
import gym
import visualize

GAME = 'CartPole-v0'
env = gym.make(GAME).unwrapped

CONFIG = "./config"
EP_STEP = 300
GENERATION_EP = 10
CHECKPOINT = 9

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ep_r = []
        for ep in range(GENERATION_EP): 
            accumulative_r = 0
            observation = env.reset()
            for t in range(EP_STEP):
                action_values = net.activate(observation)
                action = np.argmax(action_values)
                observation_, reward, done, _ = env.step(action)
                accumulative_r += reward
                if done:
                    break
                observation = observation_
            ep_r.append(accumulative_r)
        genome.fitness = np.min(ep_r)/float(EP_STEP)

def run():
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
    pop = neat.Population(config)

    # recode history
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(5))

    pop.run(eval_genomes, 10)

    # visualize training
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

def evaluation():
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
    winner = p.run(eval_genomes, 1)

    # show winner net
    node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
    visualize.draw_net(p.config, winner, True, node_names=node_names)

    net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = np.argmax(net.activate(s))
            s, r, done, _ = env.step(a)
            if done: break

if __name__ == '__main__':
    TRAINING = sys.argv[1]

    if TRAINING == 'TRAIN':
        run()
    elif TRAINING == 'EVAL':
        evaluation()
    else:
        print('Please indicate TRAIN or EVAL')
