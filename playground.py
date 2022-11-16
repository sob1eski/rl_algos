import json
import numpy as np
import multiprocessing
import torch
from datetime import datetime
from simple_policy_gradient import SPG
from vanilla_policy_gradient import VPG

def mp_func(rs):
    return {
    rs: {
        'spg': SPG(experiment_params = {
                    'epochs': 100,
                    'batch_size': 5000,
                    'display_every': 200,
                    'random_seed': rs,
                    'display': False,
                    'print': False,
                }).run(),
        'vpg': VPG(experiment_params = {
                    'epochs': 100,
                    'policy_batch_size': 5000,
                    'value_batch_size': 128,
                    'display_every': 10,
                    'random_seed': rs,
                    'discount_factor': 0.98,
                    'device': torch.device('cpu'),
                    'display': False,
                    'print': False
                }).run()
        }
    }

if __name__ == '__main__':
    n = 8
    rss = list(range(n))
    p = multiprocessing.Pool()
    results_list = p.map(mp_func, rss)
    results = {}
    for d in rss:
        results.update(d)
    with open(f'comp_{datetime.now()}.json', 'w') as f:
        json.dump(results, f)
