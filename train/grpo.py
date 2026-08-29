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

class GRPOTrainer:
    def __init__(
        self,
        org_model:MTP,
        reward_fn:callable,
        eps:float, beta:float
    )->None:
        self.old = org_model.qwen3()
        self.policy = org_model.qwen3()

        for p in self.old.parameters():
            p.requires_grad = False
        
        self.reward_fn = reward_fn

        self.eps, self.beta = eps, beta
    
    def _compute_objective(self, group:Group, prob_old:torch.Tensor|None = None)->torch.Tensor:
        sz = group.size
        sum = 0.0
        for i in range(sz):
            oi = group.outputs[i]
            ai = group.outputs[i]

            prob_cur = self._compute_prob(self.policy, oi)
            if prob_old is None:
                with torch.inference_mode():
                    prob_old = self._compute_prob(self.old, oi)
            prob_ref = prob_old

            frac = torch.exp(torch.log(prob_cur)-torch.log(prob_old))
            dkl = self._compute_dkl(prob_cur, prob_ref)

            obj = torch.min(
                frac * ai,
                torch.clip(frac, 1-self.eps, 1+self,eps)*ai
            ) - self.beta * dkl

            sum += obj
        return sum / sz

    @torch.inference_mode()
    def _sampler(self, model:Qwen3ForCausalLM, group_size:int, input:torch.Tensor)->torch.Tensor:
        pass
    
    def _compute_prob(self, model:Qwen3ForCausalLM, output:torch.Tensor)->torch.Tensor:
        pass
    
    def _compute_advantages(self, rewards:torch.Tensor)->torch.Tensor:
        return (rewards - torch.mean(rewards)) / torch.std(rewards)
    
    def _compute_dkl(self, prob_cur:torch.Tensor, prob_ref:torch.Tensor)->torch.Tensor:
        dis = torch.log(prob_ref) - torch.log(prob_cur)
        return torch.exp(dis) - dis - 1.0
    

    def loss(self, group:Group, prob_old:torch.Tensor|None = None)->torch.Tensor:
        return -self._compute_objective(group, prob_old=prob_old)

    