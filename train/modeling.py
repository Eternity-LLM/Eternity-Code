import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding, Qwen3DecoderLayer, Qwen3PreTrainedModel, Qwen3Model, Qwen3ForCausalLM
from transformers import GenerationMixin

class Embedding(nn.Embedding):
    '''
    Embedding layer with optional LoRA (Low-Rank Adaptation) support. If lora_rank is greater than 0, the embedding layer will include additional low-rank matrices for adaptation.
    
    Methods:
    - __init__(vocab_size, dim, lora_rank): Initializes the embedding layer with the given vocabulary size, embedding dimension, and optional LoRA rank.
    - forward(input_ids): Computes the embedding for the given input IDs, optionally adding the LoRA adaptation if enabled.
    - enable_lora(rank): Enables LoRA with the specified rank, freezing the original embedding weights and initializing the LoRA matrices.
    - disable_lora(): Disables LoRA, merging the LoRA adaptation into the original embedding weights and freeing the LoRA matrices.
    '''
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

        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def disable_lora(self):
        assert self.rank>0, 'LoRA is not available'
        self.rank = 0

        lora = self.lora_A @ self.lora_B
        self.weight.data += lora.data
        self.weight.requires_grad = True
        self.lora_A = None
        self.lora_B = None

class Block(Qwen3DecoderLayer):
    '''
    Eternity-Code Decoder Layer with Dropout. This class extends the Qwen3DecoderLayer to include a dropout layer after the forward pass.

    Methods:
    - __init__(config, layer_idx, dropout_rate): Initializes the Block with the given configuration, layer index, and dropout rate.
    - forward(hidden_states, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, **kwargs): Performs the forward pass through the decoder layer and applies dropout to the output hidden states.
    '''
    def __init__(self, config, layer_idx, dropout_rate:float = 0.07):
        super().__init__(config, layer_idx)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, hidden_states, attention_mask = None, position_ids = None, past_key_values = None, use_cache = False, position_embeddings = None, **kwargs):
        return self.dropout(super().forward(hidden_states, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, **kwargs))

class Model(Qwen3Model):
    '''
    Qwen3 Model with optional LoRA support. This class extends the Qwen3Model to include an embedding layer that can optionally use LoRA for low-rank adaptation.

    Methods:
    - __init__(config, dropout_rate, lora_rank): Initializes the Model with the given configuration, dropout rate, and optional LoRA rank. The embedding layer will use LoRA if lora_rank is greater than 0.
    - forward(inputs_embeds, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, **kwargs): Performs the forward pass through the model, using the embedding layer and the decoder layers to produce the output hidden states.
    '''
    def __init__(self, config:Qwen3Config, dropout_rate:float = 0.07, lora_rank:int = 0):
        Qwen3PreTrainedModel.__init__(self, config)
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
    '''
    MTP Module that processes the output of the base model and the new token embeddings for multi-token prediction.

    Methods:
    - __init__(config, layer_idx, dropout_rate): Initializes the MTPModule with the given configuration, layer index, and dropout rate.
    - forward(last, new_tok, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, **kwargs): Performs the forward pass through the MTPModule, combining the last hidden state and new token embeddings, and passing them through a linear layer and a Block for further processing.
    '''
    def __init__(self, config, layer_idx:int = 0, dropout_rate:float = 0.07):
        super().__init__()
        self.norm1 = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.block = Block(config, layer_idx, dropout_rate)

    def forward(self, last, new_tok, attention_mask = None, position_ids = None, past_key_values = None, use_cache = False, position_embeddings = None, **kwargs):
        last = self.norm1(last)
        new_tok = self.norm2(new_tok)

        hidden = self.fc(torch.cat([last, new_tok], dim=-1))
        hidden = self.block(hidden, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, **kwargs)

        return hidden

class MTP(Qwen3PreTrainedModel, GenerationMixin):
    '''
    MTP (Multi-Token Pridiction) model that extends the Qwen3 architecture to support multi-token prediction.

    Methods:
    - __init__(config, dropout_rate, lora_rank, mtp_depth): Initializes the MTP model with the given configuration, dropout rate, LoRA rank, and MTP depth.
    - forward(input_ids, attention_mask, position_ids, past_key_values, use_cache, position_embeddings, **kwargs): Performs the forward pass through the MTP model, handling both training and inference modes.
    - load_qwen3(qwen3): Loads weights from a Qwen3ForCausalLM model into the MTP model.
    - from_pretrained(*args, **kwargs): Raises NotImplementedError, indicating that MTP.from_pretrained is not implemented and suggesting to use MTP.load_qwen3(Qwen3ForCausalLM.from_pretrained(...)) instead.

    Note:
    To load weights from a pre-trained Qwen3ForCausalLM model, use the load_qwen3 method after initializing the MTP model.
    That is, you can do the following:

    ```python
    mtp_model = MTP(config, dropout_rate=0.07, lora_rank=0, mtp_depth=3)
    qwen3_model = Qwen3ForCausalLM.from_pretrained('path_to_pretrained_model')
    mtp_model.load_qwen3(qwen3_model)
    ```
    '''

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config, dropout_rate:float = 0.07, lora_rank:int = 0, mtp_depth:int = 3):
        super().__init__(config)
        self.model = Model(config, dropout_rate=dropout_rate, lora_rank=lora_rank)
        self.vocab_size = config.vocab_size

        self.emb = self.model.embed_tokens
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_depth = mtp_depth
        self.mtp_modules = nn.ModuleList([MTPModule(config, i+config.num_hidden_layers+1, dropout_rate) for i in range(mtp_depth)])

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False, position_embeddings=None, **kwargs):
        if self.training:
            assert use_cache==False and past_key_values is None, "Train mode does not support use_cache or past_key_values"

            emb = self.emb(input_ids)
            hidden = torch.cat(
                [
                    emb, 
                    torch.zeros(input_ids.shape[0], self.mtp_depth, self.config.hidden_size, device=emb.device, dtype=emb.dtype)
                ],
                dim=1
            )
            output = self.model.forward(
                inputs_embeds=hidden[:, :input_ids.shape[1], :], 
                attention_mask=attention_mask, position_ids=position_ids, 
                past_key_values=past_key_values, use_cache=use_cache, 
                position_embeddings=position_embeddings, **kwargs
            ).last_hidden_state.unsqueeze(1)

            mtp_out = output

            for i, module in enumerate(self.mtp_modules, start=1):
                mtp_out = module(
                    mtp_out.squeeze(1),
                    hidden[:, i:i+input_ids.shape[1], :],
                    attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, use_cache=use_cache,
                    position_embeddings=position_embeddings, **kwargs
                ).unsqueeze(1)
                output = torch.cat([output, mtp_out], dim=1)

            # (batch_size, mtp_depth+1, seq_len, hidden_dim)
            # to (batch_size, mtp_depth+1, seq_len, vocab_size)
            logits = self.lm_head(output)

            return CausalLMOutputWithPast(
                loss=None,
                logits=logits, # (batch_size, mtp_depth+1, seq_len, vocab_size)
                past_key_values=None,
                hidden_states=output, # (batch_size, mtp_depth+1, seq_len, hidden_dim)
                attentions=None
            )
        else:
            return Qwen3ForCausalLM.forward(self, input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)

    def load_qwen3(self, qwen3:Qwen3ForCausalLM):
        self.model.load_state_dict(qwen3.model.state_dict(), strict=False)
        self.lm_head.load_state_dict(qwen3.lm_head.state_dict(), strict=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("MTP.from_pretrained is not implemented. Use MTP.load_qwen3(Qwen3ForCausalLM.from_pretrained(...)) to load weights from a Qwen3ForCausalLM model.")
