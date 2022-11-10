import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import gymnasium as gym

defaults = {
    'env_name': 'LunarLander-v2',
    'policy_net_params': {
        'inner_sizes': [16],
        'inner_activation': nn.ReLU,
        'output_activation': nn.Identity,
    },
    'policy_optimizer_params': {
        'type': Adam,
        'lr': 1e-2
    },
    'experiment_params': {
        'epochs': 205,
        'batch_size': 5000,
        'display_every': 20,
        'random_seed': 42
    }
}

class PolicyNetwork(nn.Module):
    
    def __init__(self, sizes, inner_activation, output_activation):
        super(PolicyNetwork, self).__init__()
        self.sequential = self.set_sequential(
            sizes, inner_activation, output_activation)

    @staticmethod
    def set_sequential(sizes, activation, output_activation):
        layers = []
        for j in range(len(sizes) - 1):
            act_func = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act_func()]
        return nn.Sequential(*layers)

    @staticmethod
    def make_net(sizes, inner_activation, output_activation):
        return PolicyNetwork(sizes, inner_activation, output_activation)

    def forward(self, x):
        x = self.sequential(x)
        return x

    def get_policy(self, observation):
        logits = self(observation)
        return Categorical(logits = logits)

    def get_action(self, observation):
        return self.get_policy(observation).sample().item()

    def compute_loss(self, observation, action, weights):
        logp = self.get_policy(observation).log_prob(action)
        return - (logp * weights).mean()

class DataManager():

    def __init__(self):
        self.clear_data()

    def clear_data(self):
        names = [
            'observations',
            'actions',
            'weights',
            'returns',
            'ep_lengths'
        ]
        self.data = {name: [] for name in names}
        return None

    def set_loss(self, loss):
        self.data['loss'] = loss

    def print_epoch_stats(self, i):
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (i, self.data['loss'], np.mean(self.data['returns']), np.mean(self.data['ep_lengths'])))

    def compute_rtgs(self, rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (rtgs[i + 1] if i + 1 < n else 0)
        return rtgs
    
    def get_data_for_update(self):
        return [
            torch.as_tensor(np.array(self.data['observations']), dtype = torch.float32),
            torch.as_tensor(self.data['actions'], dtype = torch.int32),
            torch.as_tensor(self.data['weights'], dtype = torch.float32)
        ]

class SPG():

    def __init__(
        self,
        env_name = defaults['env_name'],
        policy_net_params = defaults['policy_net_params'],
        policy_optimizer_params = defaults['policy_optimizer_params'],
        experiment_params = defaults['experiment_params']
    ):  
        self.params = experiment_params
        self.env_name = env_name
        self.policy_net_params = policy_net_params
        self.policy_optimizer_params = policy_optimizer_params
        self.mng = DataManager()
        self.epochs_counter = 0
        
    def set_env(self):
        self.env = gym.make(self.env_name, render_mode = 'rgb_array')
        self.env.reset(seed = self.params['random_seed'])

    def set_policy_net(self):
        self.policy_net = PolicyNetwork.make_net(
            sizes = [self.env.observation_space.shape[0]] + \
                self.policy_net_params['inner_sizes'] + \
                    [self.env.action_space.n],
            inner_activation = self.policy_net_params['inner_activation'],
            output_activation = self.policy_net_params['output_activation']
        )

    def set_policy_optimizer(self, type, lr):
        self.policy_optimizer = type(self.policy_net.parameters(), lr = lr)

    def run_episode(self):
        observation, info = self.env.reset()
        ep_rewards = []
        while True:
            self.mng.data['observations'].append(observation)
            action = self.policy_net.get_action(
                torch.as_tensor(observation, dtype = torch.float32)
            )
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.mng.data['actions'].append(action)
            ep_rewards.append(reward)
            if terminated or truncated:
                self.mng.data['returns'].append(sum(ep_rewards))
                self.mng.data['ep_lengths'].append(len(ep_rewards))
                self.mng.data['weights'] += list(self.mng.compute_rtgs(ep_rewards))
                break
    
    def run_episode_render(self, env):
        observation, _ = env.reset()
        while True:
            action = self.policy_net.get_action(
                torch.as_tensor(observation, dtype = torch.float32)
            )
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            
    def run_batch(self):
        self.mng.clear_data()
        while len(self.mng.data['observations']) < self.params['batch_size']:
            self.run_episode()

    def update_policy(self):
        self.policy_optimizer.zero_grad()
        data_for_update = self.mng.get_data_for_update()
        batch_loss = self.policy_net.compute_loss(
            *data_for_update
        )
        batch_loss.backward()
        self.policy_optimizer.step()
        self.mng.set_loss(batch_loss)

    def render_human_env(self):
        env = gym.make(self.env_name, render_mode = 'human')
        _, _ = env.reset()
        with torch.no_grad():
            self.run_episode_render(env)
        env.close()

    def run(self):
        self.set_env() # make environment from env_name and reset it during initialization
        self.set_policy_net() # setup policy network using info from env
        self.set_policy_optimizer(**self.policy_optimizer_params) # add its optimizer
        while self.epochs_counter < self.params['epochs']:
            self.run_batch() # run episodes until batch_size is exceeded
            self.update_policy() # update policy with batch data
            self.mng.print_epoch_stats(self.epochs_counter) # print stats from this batch
            self.mng.clear_data() # clear logs buffer
            if self.epochs_counter % self.params['display_every'] == 0:
                self.render_human_env() # occassionally display agents performance 
            self.epochs_counter += 1 # epoch finished
        self.env.close()
    
if __name__ == '__main__':
    spg = SPG()
    spg.run()