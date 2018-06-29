import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from helpful_functions import *
import main

# CPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# ENV
env = main.Tetris()
report_interval = 10
quiet = False
solved = False
solved_episode = 0

# Seeds
seed_value = 2
torch.manual_seed(seed_value)
random.seed(seed_value)

# Parameters
number_of_inputs = 220
number_of_outputs = 4
num_episodes = 6000
episode_step_limit = 500
env_render_interval = 40

learning_rate = 0.01
gamma = .999

steps_total = []
rewards_total = []

update_target_frequency = 1000

batch_size = 32
replay_mem_size = 50000
hidden_layer = 128

egreedy = 0.9
egreedy_final = 0.01
egreedy_decay = 30000

agent_path = "manual_training"
load = True
double_dqn = False
clip_error = True
####################

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

class ExperienceReplay():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(number_of_inputs, hidden_layer)
        self.advantage = nn.Linear(hidden_layer, number_of_outputs)
        self.value = nn.Linear(hidden_layer, 1)
        self.activation = nn.Tanh()
        # self.activation = nn.ReLU()

    def forward(self, x):
        output1 = self.linear1(x)
        output1 = self.activation(output1)

        output_advantage = self.advantage(output1)
        output_value = self.value(output1)

        output_final = output_value + output_advantage - output_advantage.mean()
        return output_final

class QNet_Agent():
    def __init__(self):
        self.nn = NeuralNetwork().to(device)
        self.target_nn = NeuralNetwork().to(device)

        self.loss_func = nn.MSELoss()
        # loss_func = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(params=self.nn.parameters(), lr=learning_rate)
        # optimizer = optim.RMSprop(params=self.nn.parameters(), lr=learning_rate)

        self.update_target_counter = 0

    def select_action(self, state, epsilon):
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > epsilon:
            with torch.no_grad():
                state = torch.Tensor(state).to(device)
                actions_from_nn = self.nn(state)
                action = torch.max(actions_from_nn, 0)[1].item()
        else:
            action = env.sample()

        return action

    def optimize(self):
        if len(memory) < batch_size:
            return

        state, action, new_state, reward, done = memory.sample(batch_size)
        state = torch.Tensor(state).to(device)
        action = torch.LongTensor(action).to(device)
        new_state = torch.Tensor(new_state).to(device)
        reward = torch.Tensor(reward).to(device)
        done = torch.Tensor(done).to(device)

        if double_dqn:
            new_state_indexes = self.nn(new_state).detach()
            max_new_state_indexes = torch.max(new_state_indexes, 1)[1]
            new_state_values = self.nn(new_state).detach()
            max_new_state_values = new_state_values.gather(1, max_new_state_indexes.unsqueeze(1)).squeeze(1)
        else:
            new_state_values = self.target_nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values, 1)[0]

        target_value = reward + (1 - done) * gamma * max_new_state_values
        predicted_value = self.nn(state).gather(1, action.unsqueeze(1)).squeeze(1)

        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()

        if clip_error:
            for param in self.nn.parameters():
                param.grad.data.clamp(-1, 1)
        self.optimizer.step()

        if self.update_target_counter % update_target_frequency == 0:
            self.target_nn.load_state_dict(self.nn.state_dict())

        self.update_target_counter += 1

        # Q[state, action] = reward + gamma * torch.max(Q[new_state])


start_time = time.time()
qnet_agent = QNet_Agent()

if load:
    try:
        qnet_agent = torch.load(agent_path)
    except IOError:
        print("creating file: " + agent_path)
memory = ExperienceReplay(replay_mem_size)
frames_total = 0
for episode in range(num_episodes):
    state = env.reset()
    step = 0
    rewards = 0
    while True:
        frames_total += 1
        step += 1
        if load:
            epsilon = egreedy_final
        else:
            epsilon = calculate_epsilon(frames_total)

        if episode < 0:
            action = env.select_action()
        else:
            action = qnet_agent.select_action(state, epsilon)
        new_state, reward, done, info = env.step(action)
        memory.push(state, action, new_state, reward, done)
        qnet_agent.optimize()
        rewards += reward
        state = new_state
        if not done or step > episode_step_limit:
            steps_total.append(step)
            rewards_total.append(rewards)
            if sum(rewards_total[-100:])/len(rewards_total) > 100:
                solved = True
                solved_episode = episode
            if episode % 10 == 0 and not quiet:
                print("\n*** Episode {} ***".format(episode))
                print("Average Reward: [last {}]: {}".format(report_interval, sum(rewards_total[-report_interval:])/report_interval))
                print("Average Reward All: {}".format(sum(rewards_total)/len(rewards_total)))
                print("Epsilon: {}".format(epsilon))
                print("Elapsed Time: {}".format(time.strftime("%M:%S", time.gmtime(time.time()-start_time))))
            if episode % 10 == 0:
                print("saving")
                torch.save(qnet_agent, agent_path)
            break

plot = rewards_total
step_stats(steps_total)
reward_stats(rewards_total)
episodes_until_reward_above_mean(rewards_total)
if solved:
    print("Solved After {} episodes!".format(solved_episode))
basic_plot_time("Rewards Received Each Episode", rewards_total, "red")
basic_plot_time("Steps Taken Each Episode", steps_total, 'blue')
