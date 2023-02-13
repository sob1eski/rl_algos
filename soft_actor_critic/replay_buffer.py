import torch
import numpy as np
from dataclasses import dataclass

class ReplayBuffer():

    class Transition():

        def __init__(
                self,
                state,
                action,
                next_state,
                reward,
                terminated,
                truncated,
                device):
            
            self.state = torch.tensor(state)
            self.action = torch.tensor(action)
            self.next_state = torch.tensor(next_state)
            self.reward = torch.tensor(reward)
            self.terminated = torch.tensor(terminated, torch.bool)
            self.truncated = torch.tensor(truncated, torch.bool)


    def __init__(
            self,
            buffer_capacity,
            batch_size):
        
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = torch.empty(buffer_capacity, dtype = ReplayBuffer.Transition)
    
    def get_minibatch(self):
        if self.buffer.shape[0] > self.batch_size:
            print('Sampling minibatch...')
            return np.random.choice(self.buffer, size = self.batch_size)
        else:
            print('Buffer not big enough, returning false.')
            return False

if __name__ == '__main__':

    ex_args = {
        'state': [],
        'action': [],
        'next_state': [],
        'reward': torch.tensor(5),
        'terminated': False,
        'truncated': False
    }

    transition = ReplayBuffer.Transition(**ex_args)

    print(transition)