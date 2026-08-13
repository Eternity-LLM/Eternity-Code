import torch
import torch.nn as nn
import troch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass

world_size = 1
rank = 0

@dataclass
class ModelArgs:
    '''
    Qwen3-dense Model Arguments
    '''
    pass

