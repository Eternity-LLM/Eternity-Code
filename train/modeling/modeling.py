import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model, Qwen3PreTrainedModel, Qwen3RMSNorm, Qwen3RotaryEmbedding
from transformers import GenerationMixin

class Embedding(nn.Embedding):
    def __init__(self, vocab_size:int, dim:int, lora_rank:int = 0):
        assert lora_rank >= 0, "lora_rank must be non-negative"

        super().__init__(vocab_size, dim)

        self.vocab_size = vocab_size
        self.dim = dim

        self.rank = 0
        self.lora_A, self.lora_B = None, None
        self.enable_lora(lora_rank) if lora_rank > 0 else None

    def forward(self, input_ids):
        original = super().forward(input_ids)
        if self.rank > 0:
            lora = F.embedding(input_ids, self.lora_A) @ self.lora_B
            return original + lora
        return original

    def enable_lora(self, rank:int):
        assert self.rank==0, 'LoRA is already available'
        assert rank>0, f'Invalid rank {rank}'
        self.rank = rank

        for param in self.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.vocab_size, rank), requires_grad=True)
        self.lora_B = nn.Parameter(torch.empty(rank, self.dim), requires_grad=True)


class Block(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx, dropout_rate:float = 0.07):
        super().__init__(config, layer_idx)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, hidden_states, attention_mask = None, position_ids = None, past_key_values = None, use_cache = False, position_embeddings = None, **kwargs):
        return self.dropout(super().forward(hidden_states, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, **kwargs))

class Model(Qwen3PreTrainedModel, Qwen3Model):
    def __init__(self, config:Qwen3Config, dropout_rate:float = 0.07, lora_rank:int = 0):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.dropout_rate = dropout_rate
        self.lora_rank = lora_rank

        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, lora_rank)
        self.layers = nn.ModuleList([Block(config, i, dropout_rate) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

class MTPModule(nn.Module):
    def __init__(self, config):
        self.norm1 = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.block = Block(config, 0)

    def forward(self, last, new_tok):
        last = self.norm1(last)
        new_tok = self.norm2(new_tok)

        hidden = self.fc(torch.cat(last, new_tok, dim=1))
        hidden = self.block(hidden)

        return hidden

class MTP(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Model(config)
        self.vocab_size = config.vocab_size

        self.emb = self.model.embed_tokens
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward():
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs, ignore_mismatched_sizes=True)
