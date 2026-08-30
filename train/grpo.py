# GRPO Training Script for Eternity-Code MTP
from modeling.modeling import *
from typing import List, Tuple
from transformers import AutoTokenizer

import torch
import torch.nn.functional as F

class Group:
    sequences:List[torch.Tensor]=[]
    advantages:List[torch.Tensor]|List[float]=[]

    @property
    def size(self)->int:
        assert len(self.sequence)==len(self.advantages)
        return len(self.advantages)

def eager_sampler(model:Qwen3ForCausalLM, group_size:int, input_seq:torch.Tensor)->Tuple[torch.Tensor, List[torch.Tensor]]:
    pass

class GRPOTrainer:
    def __init__(
        self,
        org_model:MTP, tokenizer:AutoTokenizer,
        reward_fn:callable,
        eps:float, beta:float,
        sampler:callable|None=None
    )->None:
        self.old = org_model    # Class 'MTP'. However, `self.old` is only used in inference mode, so it's equivalent to class 'Qwen3ForCausalLM', see ./modeling/modeling.py
        self.policy = org_model.qwen3()    # Class 'Qwen3ForCausalLM', policy model
        self.tokenizer = tokenizer

        for p in self.old.parameters():
            p.requires_grad = False
        
        self.reward_fn = reward_fn

        self.eps, self.beta = eps, beta

        self.sampler_fn = sampler if sampler is not None else eager_sampler
    
    @torch.inference_mode()
    def sampler(self, group_size:int, prompt:str|dict)->Tuple[Group, torch.Tensor]:
        # Return group & _prob_old

        input_seq = None  # tokenize
        o_list = self.sampler_fn(self.old, group_size, input_seq)
        # decode & reward_fn, _compute_advantages, ...
        pass
    
    def _compute_objective(self, group:Group, _prob_old:torch.Tensor|None = None, prompt_len:int=0)->torch.Tensor:
        sz = group.size
        sum = 0.0
        for i in range(sz):
            oi = group.sequences[i]
            ai = group.advantages[i]

            prob_cur = self._compute_prob(self.policy, oi, prompt_len)
            if _prob_old is None:
                with torch.inference_mode():
                    prob_old = self._compute_prob(self.old, oi, prompt_len)
            else:
                prob_old = _prob_old[i]
            prob_ref = prob_old

            frac = torch.exp(prob_cur-prob_old)
            dkl = self._compute_dkl(prob_cur, prob_ref)

            obj = torch.min(
                frac * ai,
                torch.clip(frac, 1-self.eps, 1+self,eps)*ai
            ) - self.beta * dkl

            sum += obj
        return sum / sz

    def _compute_prob(self, model:Qwen3ForCausalLM, seq:dict|torch.Tensor, prompt_len:int)->torch.Tensor:
        if not isinstance(seq, dict):
            seq_len = seq.shape[1]
            with torch.no_grad():
                attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=seq.device))
            seq = {'input_ids':seq, 'attention_mask':attn_mask}
        
        output:CausalLMOutputWithPast = model.forward(**seq)
        logits = output.logits
        log_probs = F.log_softmax(logits, dim=-1)[:, prompt_len-1:, ...]
        token_log_probs = log_probs.gather(
            dim=-1,
            index=seq['input_ids'].unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs.sum(dim=-1)
    
    @torch.inference_mode()
    def _compute_advantages(self, rewards:torch.Tensor|List[torch.Tensor]|List[float])->List[float]:
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards)
        return ((rewards - torch.mean(rewards)) / torch.std(rewards)).numpy().tolist()
    
    def _compute_dkl(self, prob_cur:torch.Tensor, prob_ref:torch.Tensor)->torch.Tensor:
        dis = prob_ref - prob_cur
        return torch.exp(dis) - dis - 1.0
    
    def loss(self, group:Group, _prob_old:torch.Tensor|None = None, prompt_len:int=0)->torch.Tensor:
        return -self._compute_objective(group, _prob_old=_prob_old, prompt_len=prompt_len)

    