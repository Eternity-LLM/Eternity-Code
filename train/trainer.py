import torch
import torch_npu
import torch.nn as nn

from modeling.modeling import MTP

from typing import List

def expert_trainer(exp_model:MTP):
    pass

def final_trainer(final_model:MTP, exp_list:List[MTP]):
    pass

__all__ = ['expert_trainer', 'final_trainer']