import torch
import numpy as np
import random
from typing import Literal

def set_random_seed(seed=42, device = 'cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True     
        torch.backends.cudnn.benchmark = False
        
def get_anneal_ramp_param(epoch, ramp_start, ramp_end, max_param, method: Literal['cosine', 'linear'] = 'linear'):
    t = np.clip((epoch - ramp_start) / (ramp_end - ramp_start), a_min = 0, a_max = 1)
    if method == 'linear':
        return max_param * t
    elif method == 'cosine':
        return max_param * 1/2 * (1 - np.cos(np.pi * t))
    else:
        raise ValueError(f"Annealing method must be one of ['cosine', 'linear']. Got {method}")

