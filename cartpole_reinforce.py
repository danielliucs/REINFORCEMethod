import nntplib
import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        #neural network!!
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

#accepts a list of rewards
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards): #since last step of episode has total reward equal to local
        sum_r *= GAMMA #gamma * r_t
        sum_r += r #added with r_t-1
        res.append(sum_r)
    return list(reversed(res))

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    #internally calls the random.choice from NumPy with probabilities from network
    #convert network output to probabilities by calling softmax first
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                   apply_softmax=True) #makes a decision about actions for every observation
    #returns experience tuples
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    #contains toward rewards for episodes and coutn of completed episodes
    total_rewards = []
    step_idx = 0
    done_episodes = 0

    #gather training data
    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = [] #local rewards

    #beginning of training loop
    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        # handle new rewards at the end of the episode, also responsible for tensorboard data
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break
        #if enough episodes are trained, we perform optimization in the gathered examples
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        #convert to tensors
        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        #calculate loss from steps, calclate states into logits and log + softmax them
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)

        #log probabilities
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        
        #average scaled values and negate to obtain loss to minimize, minus sign important!
        loss_v = -log_prob_actions_v.mean()

        #classic part of training algorithm
        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
