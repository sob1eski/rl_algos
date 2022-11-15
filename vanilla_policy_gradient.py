import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import gymnasium as gym

defaults = {
    'env_name': 'LunarLander-v2',
    'policy_net_params': {
        'inner_sizes': [32],
        'inner_activation': nn.ReLU,
        'output_activation': nn.Identity
    },
    'policy_optimizer_params': {
        'type': Adam,
        'lr': 1e-2
    },
    'value_net_params': {
        'inner_sizes': [32],
        'inner_activation': nn.ReLU,
        'output_activation': nn.Identity,
        'lambda_factor': 0.96 # 0.96
    },
    'value_optimizer_params': {
        'type': Adam,
        'lr': 1e-2
    },
    'experiment_params': {
        'epochs': 200,
        'policy_batch_size': 5000,
        'value_batch_size': 128,
        'display_every': 10,
        'random_seed': 42,
        'discount_factor':  0.98, # 0.98
        'device': torch.device('cpu')
    }
}

class PolicyNetwork(nn.Module):
    
    def __init__(self, sizes, inner_activation, output_activation, optimizer, lr, data_manager):
        super(PolicyNetwork, self).__init__()
        self.sequential = self.set_sequential(
            sizes, inner_activation, output_activation)
        self.optimizer = optimizer(self.parameters(), lr = lr)
        self.mng = data_manager

    def set_sequential(self, sizes, activation, output_activation):
        layers = []
        for j in range(len(sizes) - 1):
            act_func = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act_func()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sequential(x)
        return x

    def get_policy(self, observation):
        logits = self(torch.as_tensor(observation, dtype = torch.float32))
        return Categorical(logits = logits)

    def get_action(self, observation):
        return self.get_policy(observation).sample().item()

    def compute_loss(self, observation, action, weights):
        logp = self.get_policy(observation).log_prob(action)
        return - (logp * weights).mean()

    def update_weights(self):
        self.optimizer.zero_grad()
        observations, actions, _, weights = self.mng.get_data_for_update()
        batch_loss = self.compute_loss(
            observations,
            actions,
            weights
        )
        batch_loss.backward()
        self.optimizer.step()
        self.mng.set_loss(batch_loss, type = 'policy')

class ValueNetwork(nn.Module):

    def __init__(
        self,
        sizes,
        inner_activation,
        output_activation,
        optimizer,
        lr,
        batch_size,
        discount_factor,
        lambda_factor,
        data_manager
    ):
        super(ValueNetwork, self).__init__()
        self.sequential = self.set_sequential(
            sizes, inner_activation, output_activation)
        self.loss = nn.MSELoss()
        self.optimizer = optimizer(self.parameters(), lr = lr)
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.lambda_factor = lambda_factor
        self.mng = data_manager

    def set_sequential(self, sizes, activation, output_activation):
        layers = []
        for j in range(len(sizes) - 1):
            act_func = activation if j < len(sizes) - 2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act_func()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sequential(x)
        return x

    def get_delta(self, timestep, value_estims, rewards):
        return rewards[timestep] + self.discount_factor * \
            value_estims[timestep + 1] - value_estims[timestep]

    def get_gae(self, observations, rewards):
        gae = np.zeros_like(rewards)
        observations = torch.as_tensor(np.array(observations), dtype = torch.float32)
        rewards = torch.as_tensor(rewards)
        with torch.no_grad():
            value_estims = self(observations)
        ep_len = len(rewards)
        for i, timestep in enumerate(range(ep_len)):
            discounted_deltas = [(self.discount_factor * self.lambda_factor) ** l * \
                self.get_delta(l, value_estims, rewards) for\
                l in range(timestep, ep_len - 1)]
            gae[i] = sum(discounted_deltas)
        return gae

    def update_weights(self):
        observations, _, rtgs, _ = self.mng.get_data_for_update()
        rtgs = torch.as_tensor(rtgs).unsqueeze(-1)
        batches_idxs = torch.randperm(observations.size(0))
        i = 0
        idxs = batches_idxs[i * self.batch_size : (i+1) * self.batch_size]
        while idxs.size(0) > 0:
            batch_obs = observations[idxs]
            batch_values = self(batch_obs)
            batch_rtgs = rtgs[idxs]
            self.optimizer.zero_grad()
            batch_loss = self.loss(
                batch_values,
                batch_rtgs
            )
            batch_loss.backward()
            self.optimizer.step()
            self.mng.set_loss(batch_loss, type = 'value')
            i += 1
            idxs = batches_idxs[i * self.batch_size : (i+1) * self.batch_size]

class DataManager():

    def __init__(self):
        self.clear_data()
        self.gen_data = {
            'returns': [],
            'ep_lengths': [],
            'policy_losses': [],
            'value_losses': [],
        }

    def clear_data(self):
        names = [
            'observations',
            'actions',
            'rewards',
            'rtgs',
            'weights',
            'returns',
            'ep_lengths'
        ]
        self.data = {name: [] for name in names}
        return None

    def set_loss(self, loss, type):
        if type == 'policy':
            self.gen_data['policy_losses'].append(loss.item())
            self.data['loss'] = loss
        elif type == 'value':
            self.gen_data['value_losses'].append(loss.item())

    def print_epoch_stats(self, i):
        mean_returns = np.mean(self.data['returns'])
        mean_ep_lens = np.mean(self.data['ep_lengths'])
        self.gen_data['returns'].append(mean_returns)
        self.gen_data['ep_lengths'].append(mean_ep_lens)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
            (i, self.data['loss'], mean_returns, mean_ep_lens))

    # def compute_rtgs(self, rewards):
    #     n = len(rewards)
    #     rtgs = np.zeros_like(rewards)
    #     for i in reversed(range(n)):
    #         rtgs[i] = rewards[i] + (self.discount_factor * rtgs[i + 1] if i + 1 < n else 0)
    #     return rtgs
    
    def get_data_for_update(self):
        return [
            torch.as_tensor(np.array(self.data['observations']), dtype = torch.float32),
            torch.as_tensor(self.data['actions'], dtype = torch.int32),
            torch.as_tensor(self.data['rtgs'], dtype = torch.float32),
            torch.as_tensor(self.data['weights'], dtype = torch.float32)
        ]

class VPG():

    def __init__(
        self,
        env_name = defaults['env_name'],
        policy_net_params = defaults['policy_net_params'],
        policy_optimizer_params = defaults['policy_optimizer_params'],
        value_net_params = defaults['value_net_params'],
        value_optimizer_params = defaults['value_optimizer_params'],
        experiment_params = defaults['experiment_params'],
        data_manager = DataManager()
    ):  
        self.params = experiment_params
        self.env_name = env_name
        self.policy_net_params = policy_net_params
        self.policy_optimizer_params = policy_optimizer_params
        self.value_net_params = value_net_params
        self.value_optimizer_params = value_optimizer_params
        self.mng = data_manager
        self.epochs_counter = 0
        
    def set_env(self):
        self.env = gym.make(self.env_name, render_mode = 'rgb_array')
        self.env.reset(seed = self.params['random_seed'])

    def set_policy_net(self):
        self.policy_net = PolicyNetwork(
            sizes = [self.env.observation_space.shape[0]] + \
                self.policy_net_params['inner_sizes'] + \
                    [self.env.action_space.n],
            inner_activation = self.policy_net_params['inner_activation'],
            output_activation = self.policy_net_params['output_activation'],
            optimizer = self.policy_optimizer_params['type'],
            lr = self.policy_optimizer_params['lr'],
            data_manager = self.mng
        )

    def set_value_net(self):
        self.value_net = ValueNetwork(
            sizes = [self.env.observation_space.shape[0]] + \
                self.value_net_params['inner_sizes'] + [1],
            inner_activation = self.value_net_params['inner_activation'],
            output_activation = self.value_net_params['output_activation'],
            optimizer = self.value_optimizer_params['type'],
            lr = self.value_optimizer_params['lr'],
            batch_size = self.params['value_batch_size'],
            discount_factor = self.params['discount_factor'],
            lambda_factor = self.value_net_params['lambda_factor'],
            data_manager = self.mng
        )

    def compute_rtgs(self, rewards):
        n = len(rewards)
        rtgs = np.zeros_like(rewards)
        for i in reversed(range(n)):
            rtgs[i] = rewards[i] + (self.params['discount_factor'] * rtgs[i + 1] if i + 1 < n else 0)
        return rtgs

    def run_batch(self):
        self.mng.clear_data()
        while len(self.mng.data['observations']) < self.params['policy_batch_size']:
            self.run_episode()

    def run_episode(self):
        observation, _ = self.env.reset()
        ep_rewards = []
        ep_observations = [observation]
        while True:
            self.mng.data['observations'].append(observation)
            action = self.policy_net.get_action(torch.as_tensor(observation, dtype = torch.float32))
            observation, reward, terminated, truncated, _ = self.env.step(action)
            self.mng.data['actions'].append(action)
            ep_rewards.append(reward)
            ep_observations.append(observation)
            if terminated or truncated:
                self.mng.data['returns'].append(sum(ep_rewards))
                self.mng.data['ep_lengths'].append(len(ep_rewards))
                self.mng.data['rtgs'] += list(self.compute_rtgs(ep_rewards))
                self.mng.data['weights'] += list(self.value_net.get_gae(ep_observations, ep_rewards))
                break  

    def run_episode_render(self, env):
        observation, _ = env.reset()
        while True:
            action = self.policy_net.get_action(
                observation
            )
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            
    def render_human_env(self):
        env = gym.make(self.env_name, render_mode = 'human')
        _, _ = env.reset()
        with torch.no_grad():
            self.run_episode_render(env)
        env.close()

    def run(self):
        self.set_env() # make environment from env_name and reset it during initialization
        self.set_policy_net() # setup policy network using info from env 
        self.set_value_net() # setup value network using info from env
        while self.epochs_counter < self.params['epochs']:
            self.run_batch() # run episodes until batch_size is exceeded
            self.policy_net.update_weights() # update policy with batch data
            self.value_net.update_weights() # update value net with batch data
            self.mng.print_epoch_stats(self.epochs_counter) # print stats from this batch
            self.mng.clear_data() # clear logs buffer
            if self.epochs_counter % (self.params['display_every'] - 1) == 0  and self.epochs_counter != 0:
                self.render_human_env() # occassionally display agents performance 
            self.epochs_counter += 1 # epoch finished
        self.env.close()
        print('Finished')
        return self.mng.gen_data
    
if __name__ == '__main__':
    vpg = VPG()
    vpg.run()

    # to do:
    #done # - change baseline to gae
    #done # - instead of updating value net on the same batch size as policy net, try updating with typical batch size (~128)
    #done # - transfer update_net func from algorithm to network, make dataManager an attribute of net
    # - add discounted reward