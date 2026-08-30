# GRPO Training Script for Eternity-Code MTP
from modeling.modeling import MTP, Qwen3ForCausalLM
from typing import List
import torch

class Group:
    outputs:List[torch.Tensor]=[]
    advantages:List[torch.Tensor]|List[float]=[]

    @property
    def size(self)->int:
        assert len(self.outputs)==len(self.advantages)
        return len(self.advantages)

def eager_sampler(model:Qwen3ForCausalLM, group_size:int, input:torch.Tensor)->List[torch.Tensor]:
    pass

class GRPOTrainer:
    def __init__(
        self,
        org_model:MTP,
        reward_fn:callable,
        eps:float, beta:float,
        sampler:callable|None=None
    )->None:
        self.old = org_model.qwen3()
        self.policy = org_model.qwen3()

        for p in self.old.parameters():
            p.requires_grad = False
        
        self.reward_fn = reward_fn

        self.eps, self.beta = eps, beta

        self.sampler_fn = sampler if sampler is not None else eager_sampler
    
    @torch.inference_mode()
    def sampler(self, group_size:int, input:torch.Tensor)->List[Group, torch.Tensor]:
        # Return group & _prob_old
        o_list = self.sampler_fn(self.old, group_size, input)
        # decode & reward_fn, _compute_advantages, ...
        pass
    
    def _compute_objective(self, group:Group, _prob_old:torch.Tensor|None = None)->torch.Tensor:
        sz = group.size
        sum = 0.0
        for i in range(sz):
            oi = group.outputs[i]
            ai = group.outputs[i]

            prob_cur = self._compute_prob(self.policy, oi)
            if _prob_old is None:
                with torch.inference_mode():
                    prob_old = self._compute_prob(self.old, oi)
            else:
                prob_old = _prob_old[i]
            prob_ref = prob_old

            frac = torch.exp(torch.log(prob_cur)-torch.log(prob_old))
            dkl = self._compute_dkl(prob_cur, prob_ref)

            obj = torch.min(
                frac * ai,
                torch.clip(frac, 1-self.eps, 1+self,eps)*ai
            ) - self.beta * dkl

            sum += obj
        return sum / sz

    def _compute_prob(self, model:Qwen3ForCausalLM, output:torch.Tensor)->torch.Tensor:
        pass
    
    @torch.inference_mode()
    def _compute_advantages(self, rewards:torch.Tensor|List[torch.Tensor]|List[float])->List[float]:
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards)
        return ((rewards - torch.mean(rewards)) / torch.std(rewards)).numpy().tolist()
    
    def _compute_dkl(self, prob_cur:torch.Tensor, prob_ref:torch.Tensor)->torch.Tensor:
        dis = torch.log(prob_ref) - torch.log(prob_cur)
        return torch.exp(dis) - dis - 1.0
    
    def loss(self, group:Group, _prob_old:torch.Tensor|None = None)->torch.Tensor:
        return -self._compute_objective(group, _prob_old=_prob_old)

    