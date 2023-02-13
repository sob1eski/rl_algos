import torch
import numpy as np
import gymnasium as gym

from replay_buffer import ReplayBuffer
from policy_network import PolicyNetwork
from value_network import ValueNetwork

class SAC():

    def __init__(
            self,
            config,
            *args,
            **kwargs):

        self.seed = config['seed']
        self.device = config['device']
        self.env = config['env']
        self.policy = config['policy_net']
        self.value_networks = config['value_nets']
        self.replay_buffer = config['replay_buffer']

    ## environment ##
    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env_params):
        self._env = gym.make(**env_params)
        self._env.reset(seed = self.seed)
    
    ## policy network ##
    @property
    def policy_net(self):
        return self._policy
    
    @policy_net.setter
    def policy_net(self, policy_net_params):
        pass

    ## value networks ##
    @property
    def value_nets(self):
        return self._value_nets
    
    @value_nets.setter
    def value_nets(self, value_nets_params):
        pass

    ## replay buffer ##
    @property
    def replay_buffer(self):
        return self._replay_buffer
    
    @replay_buffer.setter
    def replay_buffer(self, replay_buffer_params):
        self.replay_buffer = ReplayBuffer(seed = self.seed, **replay_buffer_params)
        pass

    ## algorithm-specific ##
    def _set_seeds(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def run(self):
        self._set_seeds()
        

if __name__ == '__main__':

    config = {
        'seed': 0,
        'env': {
            'id': 'MountainCar-v0', 
            'render_mode': 'rgb_array'},
        'policy_net': {},
        'value_nets': {},
        'replay_buffer': {}}
        
    sac = SAC(config)





        

    