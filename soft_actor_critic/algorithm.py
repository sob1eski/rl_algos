import torch
import numpy as np
import gymnasium as gym

from replay_buffer import ReplayBuffer
from policy_network import PolicyNetwork
from q_value_network import QValueNetwork

class SAC():

    def __init__(
            self,
            config,
            *args,
            **kwargs):

        self.general_params = config['general_params']
        self.additional_params = config['additional_params']
        self._set_seeds() # set seeds before initializing the networks
        self.env = config['env']
        self.q_value_nets = config['q_value_nets']
        self.policy_net = PolicyNetwork(**config['policy_net'])
        self.replay_buffer = ReplayBuffer(seed = self.additional_params['seed'], **config['replay_buffer'])

    ## environment ##
    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, env_params):
        self._env = gym.make(**env_params)
        self._env.reset(seed = self.additional_params['seed'])

    ## value networks ##
    @property
    def q_value_nets(self):
        return self._q_value_nets
    
    @q_value_nets.setter
    def q_value_nets(self, q_value_nets_params):
        pass

    ## algorithm-specific ##
    def _set_seeds(self):
        np.random.seed(self.additional_params['seed'])
        torch.manual_seed(self.additional_params['seed'])

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





        

    