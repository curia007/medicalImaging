"""
LLaMA-based Architecture for medical_imaging_01
==============================================
Implements the core LLaMA model structure for medical imaging clinical reasoning:
- RoPE (Rotary Position Embeddings)
- RMSNorm pre-normalization
- SwiGLU Gated Feed-Forward Networks
- Multi-Head Causal Self-Attention with KV cache support
- Contextual sequence embedding extraction
- Autoregressive clinical text generation
"""

import os
import json
import math
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Union, Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from .tokenizer import MedicalImaging01Tokenizer, DEFAULT_VOCAB_SIZE


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class MedicalImaging01Config:
    model_type: str = "llama"
    model_name: str = "medical_imaging_01"
    architectures: List[str] = field(default_factory=lambda: ["MedicalImaging01ForCausalLM", "LlamaForCausalLM"])
    vocab_size: int = DEFAULT_VOCAB_SIZE
    # Standard LLaMA / Hugging Face / MLX-LM parameters
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    # Aliases for backward compatibility
    hidden_dim: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    num_layers: Optional[int] = None
    intermediate_dim: Optional[int] = None
    max_seq_len: Optional[int] = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    domain: str = "medical_imaging"
    description: str = "LLaMA-based model for medical imaging diagnosis and structured clinical reasoning"

    def __post_init__(self):
        # 1. hidden dimension (default 256)
        if self.hidden_dim is not None:
            self.hidden_size = self.hidden_dim
        elif self.hidden_size is not None:
            self.hidden_dim = self.hidden_size
        else:
            self.hidden_size = 256
            self.hidden_dim = 256

        # 2. layers (default 4)
        if self.num_layers is not None:
            self.num_hidden_layers = self.num_layers
        elif self.num_hidden_layers is not None:
            self.num_layers = self.num_hidden_layers
        else:
            self.num_hidden_layers = 4
            self.num_layers = 4

        # 3. attention heads (default 8)
        if self.num_heads is not None:
            self.num_attention_heads = self.num_heads
        elif self.num_attention_heads is not None:
            self.num_heads = self.num_attention_heads
        else:
            self.num_attention_heads = 8
            self.num_heads = 8

        # 4. kv heads (default to attention heads)
        if self.num_kv_heads is not None:
            self.num_key_value_heads = self.num_kv_heads
        elif self.num_key_value_heads is not None:
            self.num_kv_heads = self.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads
            self.num_kv_heads = self.num_heads

        # 5. intermediate size (default computed from hidden_dim)
        if self.intermediate_dim is not None:
            self.intermediate_size = self.intermediate_dim
        elif self.intermediate_size is not None:
            self.intermediate_dim = self.intermediate_size
        else:
            hidden_mult = int(2 * (4 * self.hidden_dim) / 3)
            self.intermediate_dim = ((hidden_mult + 63) // 64) * 64
            self.intermediate_size = self.intermediate_dim

        # 6. max sequence length (default 512)
        if self.max_seq_len is not None:
            self.max_position_embeddings = self.max_seq_len
        elif self.max_position_embeddings is not None:
            self.max_seq_len = self.max_position_embeddings
        else:
            self.max_position_embeddings = 512
            self.max_seq_len = 512

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalImaging01Config":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        # Fallback aliases if only one format was present in the dictionary
        if "hidden_dim" in data and "hidden_size" not in data:
            filtered["hidden_size"] = data["hidden_dim"]
        elif "hidden_size" in data and "hidden_dim" not in data:
            filtered["hidden_dim"] = data["hidden_size"]

        if "num_layers" in data and "num_hidden_layers" not in data:
            filtered["num_hidden_layers"] = data["num_layers"]
        elif "num_hidden_layers" in data and "num_layers" not in data:
            filtered["num_layers"] = data["num_hidden_layers"]

        if "num_heads" in data and "num_attention_heads" not in data:
            filtered["num_attention_heads"] = data["num_heads"]
        elif "num_attention_heads" in data and "num_heads" not in data:
            filtered["num_heads"] = data["num_attention_heads"]

        if "intermediate_dim" in data and "intermediate_size" not in data:
            filtered["intermediate_size"] = data["intermediate_dim"]
        elif "intermediate_size" in data and "intermediate_dim" not in data:
            filtered["intermediate_dim"] = data["intermediate_size"]

        if "num_kv_heads" in data and "num_key_value_heads" not in data:
            filtered["num_key_value_heads"] = data["num_kv_heads"]
        elif "num_key_value_heads" in data and "num_kv_heads" not in data:
            filtered["num_kv_heads"] = data["num_key_value_heads"]

        if "max_seq_len" in data and "max_position_embeddings" not in data:
            filtered["max_position_embeddings"] = data["max_seq_len"]
        elif "max_position_embeddings" in data and "max_seq_len" not in data:
            filtered["max_seq_len"] = data["max_position_embeddings"]

        return cls(**filtered)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MedicalImaging01Config":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =====================================================================
# LLaMA Architecture Primitives
# =====================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        norm_x = x * mx.rsqrt(variance + self.eps)
        return norm_x * self.weight


class SwiGLUMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network: down_proj(silu(gate_proj(x)) * up_proj(x)).
    """
    def __init__(self, config: MedicalImaging01Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalAttention(nn.Module):
    """
    Multi-Head / Grouped Query Causal Attention with Rotary Position Embeddings (RoPE).
    """
    def __init__(self, config: MedicalImaging01Config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_dim, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_dim, bias=False)

        self.rope = nn.RoPE(dims=self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim)

        offset = cache[0].shape[1] if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache is not None:
            prev_k, prev_v = cache
            keys = mx.concatenate([prev_k, keys], axis=1)
            values = mx.concatenate([prev_v, values], axis=1)
            new_cache = (keys, values)
        else:
            new_cache = None

        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeats, axis=2)
            values = mx.repeat(values, repeats, axis=2)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attention = nn.softmax(scores, axis=-1)
        output = attention @ values

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), new_cache


class DecoderLayer(nn.Module):
    """
    LLaMA Transformer Decoder Layer.
    """
    def __init__(self, config: MedicalImaging01Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.self_attn = CausalAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        norm_x = self.input_layernorm(x)
        attn_out, new_cache = self.self_attn(norm_x, mask=mask, cache=cache)
        x = x + attn_out

        norm_x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x, new_cache


class MedicalImaging01Model(nn.Module):
    """
    Transformer backbone for medical_imaging_01.
    """
    def __init__(self, config: MedicalImaging01Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = [DecoderLayer(config) for _ in range(config.num_layers)]
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        inputs_embeds: Optional[mx.array] = None
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(inputs)

        B, L, _ = h.shape

        if mask is None and L > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            mask = mask.astype(h.dtype)

        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, new_layer_cache = layer(h, mask=mask, cache=layer_cache)
            if new_layer_cache is not None:
                new_caches.append(new_layer_cache)

        h = self.norm(h)
        return h, (new_caches if new_caches else None)


class MedicalImaging01ForCausalLM(nn.Module):
    """
    medical_imaging_01 Causal Language Model with LM projection head.
    """
    def __init__(self, config: MedicalImaging01Config):
        super().__init__()
        self.config = config
        self.model = MedicalImaging01Model(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def __call__(
        self,
        token_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
        inputs_embeds: Optional[mx.array] = None
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        hidden_states, new_cache = self.model(
            inputs=token_ids,
            mask=mask,
            cache=cache,
            inputs_embeds=inputs_embeds
        )
        logits = self.lm_head(hidden_states)
        return logits, new_cache

    def get_sequence_embedding(self, token_ids: mx.array) -> mx.array:
        """
        Extracts L2-normalized contextual sequence embedding for retrieval/matching.
        """
        hidden_states, _ = self.model(token_ids)
        mean_pooled = mx.mean(hidden_states, axis=1)
        norm = mx.sqrt(mx.sum(mx.square(mean_pooled), axis=-1, keepdims=True) + 1e-8)
        return mean_pooled / norm

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        eos_token_id: int = 2
    ) -> List[int]:
        """
        Autoregressively generates token sequence.
        """
        tokens = list(prompt_ids)
        input_array = mx.array([tokens])

        for _ in range(max_new_tokens):
            if input_array.shape[1] >= self.config.max_seq_len:
                break

            logits, _ = self(input_array)
            next_token_logits = logits[0, -1, :]

            if temperature > 0:
                scaled = next_token_logits / temperature
                next_token = int(mx.random.categorical(scaled[None, :])[0])
            else:
                next_token = int(mx.argmax(next_token_logits, axis=-1))

            tokens.append(next_token)
            if next_token == eos_token_id:
                break

            input_array = mx.array([tokens])

        return tokens


# =====================================================================
# Helper Functions & Model Wrapper
# =====================================================================

def tree_flatten(params: Any, prefix: str = "") -> List[Tuple[str, mx.array]]:
    flattened = []
    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            flattened.extend(tree_flatten(v, new_prefix))
    elif isinstance(params, list):
        for i, v in enumerate(params):
            new_prefix = f"{prefix}.{i}" if prefix else str(i)
            flattened.extend(tree_flatten(v, new_prefix))
    elif isinstance(params, mx.array):
        flattened.append((prefix, params))
    return flattened


class MedicalImaging01:
    """
    High-level API for initializing, loading, training, and running
    the medical_imaging_01 LLaMA model structure.
    """
    def __init__(
        self,
        config: Optional[MedicalImaging01Config] = None,
        model_dir: Optional[str] = None
    ):
        if model_dir and os.path.isdir(model_dir):
            cfg_path = os.path.join(model_dir, "config.json")
            if os.path.exists(cfg_path):
                self.config = MedicalImaging01Config.load(cfg_path)
            else:
                self.config = config or MedicalImaging01Config()
        else:
            self.config = config or MedicalImaging01Config()

        vocab_path = os.path.join(model_dir, "vocab.json") if model_dir else None
        self.tokenizer = MedicalImaging01Tokenizer(
            vocab_file=vocab_path if (vocab_path and os.path.exists(vocab_path)) else None,
            max_vocab_size=self.config.vocab_size
        )
        self.config.vocab_size = self.tokenizer.vocab_size
        self.model = MedicalImaging01ForCausalLM(self.config)

        if model_dir and os.path.isdir(model_dir):
            weight_path = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(weight_path):
                weight_path = os.path.join(model_dir, "weights.safetensors")
            if os.path.exists(weight_path):
                self.load_weights(weight_path)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "MedicalImaging01":
        """
        Loads the medical_imaging_01 model and tokenizer from the specified directory.
        """
        return cls(model_dir=model_dir)

    def save_pretrained(self, model_dir: str):
        """
        Saves model weights, configuration, and vocabulary in the model directory.
        """
        os.makedirs(model_dir, exist_ok=True)
        # 1. Config
        self.config.save(os.path.join(model_dir, "config.json"))
        # 2. Tokenizer vocab & configs
        self.tokenizer.save(os.path.join(model_dir, "vocab.json"))
        self.tokenizer.save_pretrained(model_dir)
        # 3. Model weights (both model.safetensors and weights.safetensors for maximum compatibility)
        mx.eval(self.model.parameters())
        flat_params = dict(tree_flatten(self.model.parameters()))
        mx.save_safetensors(os.path.join(model_dir, "model.safetensors"), flat_params)
        mx.save_safetensors(os.path.join(model_dir, "weights.safetensors"), flat_params)

    def load_weights(self, weights_path: str):
        """
        Loads safetensors weights into the model.
        """
        if os.path.exists(weights_path):
            weights = mx.load(weights_path)
            if isinstance(weights, dict):
                unflattened = tree_unflatten(list(weights.items()))
                self.model.update(unflattened)
            else:
                self.model.update(weights)

    def generate_clinical_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0
    ) -> str:
        """
        Generates clinical impression / findings text from a prompt.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if prompt_ids[-1] == self.tokenizer.token_to_id["<eos>"]:
            prompt_ids = prompt_ids[:-1]

        generated_ids = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=self.tokenizer.token_to_id["<eos>"]
        )
        return self.tokenizer.decode(generated_ids[len(prompt_ids):])
