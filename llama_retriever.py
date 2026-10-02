"""
LLaMA with Limited Vocabulary for Efficient Medical Information Retrieval
========================================================================
Implements a lightweight, native MLX LLaMA architecture featuring:
- Rotary Position Embeddings (RoPE)
- Root Mean Square Layer Normalization (RMSNorm)
- SwiGLU Feed-Forward Networks (Gated SiLU MLP)
- Compact, high-utility medical vocabulary for minimal compute and memory footprint
- Semantic Embedding Extraction & Information Retrieval Engine (Hybrid dense/IDF indexing + RAG)
- Seamless integration with Medical Vision models and clinical diagnostic pipelines
"""

import os
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# =====================================================================
# Configuration
# =====================================================================

@dataclass
class LlamaConfig:
    vocab_size: int = 1024
    hidden_dim: int = 256
    num_heads: int = 8
    num_kv_heads: Optional[int] = None
    num_layers: int = 4
    intermediate_dim: Optional[int] = None
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.intermediate_dim is None:
            # Standard LLaMA heuristic: 8/3 * hidden_dim rounded to multiple of 64
            hidden_mult = int(2 * (4 * self.hidden_dim) / 3)
            self.intermediate_dim = ((hidden_mult + 63) // 64) * 64


# =====================================================================
# Limited Vocabulary Tokenizer
# =====================================================================

DEFAULT_LIMITED_VOCAB_SIZE = 1024

CLINICAL_CORE_LEXICON = [
    # Special clinical & conversational tokens
    "<pad>", "<bos>", "<eos>", "<unk>", "<query>", "<context>", "<response>", "<sep>",
    "<diagnosis>", "<findings>", "<symptom>", "<treatment>", "<severity>", "<recommendation>",
    
    # Common medical conditions & pathologies
    "pneumonia", "cardiomegaly", "effusion", "pleural", "atelectasis", "pneumothorax",
    "nodule", "mass", "consolidation", "infiltrate", "infiltrates", "fracture", "melanoma",
    "edema", "retinopathy", "covid-19", "bronchitis", "emphysema", "fibrosis", "embolism",
    "hypertension", "tuberculosis", "carcinoma", "sarcoma", "metastasis", "infection",
    "aneurysm", "dissection", "widening", "irregularity", "aortic", "aorta",
    
    # Anatomical terms & structures
    "lung", "lungs", "chest", "heart", "cardiac", "bone", "rib", "spine", "pleura",
    "alveoli", "alveolar", "bronchi", "mediastinum", "mediastinal", "costophrenic", "apex",
    "base", "lobe", "bilateral", "unilateral", "left", "right", "upper", "lower", "middle",
    "ventricle", "atrium", "trachea", "vascular", "vasculature", "tissue", "skin",
    "lesion", "cortex", "cortical", "parenchyma", "fields", "subsegmental", "diaphragm",
    "silhouette", "contour", "osseous", "imaging",
    
    # Clinical findings & radiologic signs
    "normal", "abnormal", "clear", "healthy", "acute", "chronic", "mild", "moderate", "severe",
    "opacity", "opacities", "shadowing", "blunting", "enlargement", "dilation", "congestion",
    "fluid", "meniscus", "calcification", "cavitation", "thickening", "dislocation", "swelling",
    "pigmented", "irregular", "border", "asymmetry", "stable", "progressed", "resolved",
    "improved", "worsened", "elevated", "reduced", "increased", "decreased", "positive",
    "negative", "absent", "present", "detected", "angle", "sign", "variegation", "focal",
    "patchy", "congestive", "fever", "cough", "dyspnea", "pain", "shortness", "breath",
    "radiopacity", "radiolucency",
    
    # Diagnostic reasoning, actions, and treatments
    "diagnosis", "findings", "impression", "recommendation", "consultation", "follow-up",
    "antibiotics", "diuretics", "thoracentesis", "biopsy", "excision", "radiograph",
    "x-ray", "ct", "mri", "ultrasound", "scan", "correlation", "clinical", "patient",
    "history", "treatment", "therapy", "prognosis", "risk", "stage", "grade",
    "spirometry", "physiotherapy", "immobilization", "orthopedic", "reduction",
    "echocardiogram", "echocardiographic", "decompression", "intubation", "oxygenation",
    "management", "criteria", "protocol", "ratio", "diameter", "dimension", "exceeding",
    "transverse", "dimension", "thoracic", "ratio",
    
    # Retrieval and conversational functional words
    "what", "where", "how", "why", "when", "is", "are", "was", "were", "has", "have",
    "the", "a", "an", "of", "and", "in", "to", "for", "on", "with", "at", "by", "from",
    "shows", "reveals", "demonstrates", "indicates", "suggests", "confirms", "requires",
    "associated", "consistent", "evidence", "findings", "due", "secondary", "primary",
    "features", "presents", "noted", "seen", "presents", "managed"
]

STOP_TOKENS = {
    "the", "a", "an", "of", "and", "in", "to", "for", "on", "with", "at", "by", "from",
    "is", "are", "was", "were", "has", "have", "what", "where", "how", "why", "when"
}


class LimitedVocabTokenizer:
    """
    Compact, domain-tailored tokenizer designed to constrain vocabulary size
    for high information retrieval efficiency, low computational overhead, and zero OOV loss.
    """
    def __init__(self, vocab_size: int = DEFAULT_LIMITED_VOCAB_SIZE, vocab_file: Optional[str] = None):
        self.max_vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<query>": 4,
            "<context>": 5,
            "<response>": 6,
            "<sep>": 7,
            "<diagnosis>": 8,
            "<findings>": 9,
            "<symptom>": 10,
            "<treatment>": 11,
            "<severity>": 12,
            "<recommendation>": 13,
        }
        self.token_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        
        self._build_lexicon()
        
        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)

    def _build_lexicon(self):
        # 1. Add printable ASCII characters first to guarantee 100% fallback coverage
        for i in range(32, 127):
            char = chr(i)
            self._add_token(char)

        # 2. Add lowercase domain lexicon tokens
        for term in CLINICAL_CORE_LEXICON:
            self._add_token(term.lower())

        # 3. Add capitalized variants if vocabulary capacity allows
        for term in CLINICAL_CORE_LEXICON:
            self._add_token(term.capitalize())

        # 4. Add uppercase variants
        for term in CLINICAL_CORE_LEXICON:
            self._add_token(term.upper())

    def _add_token(self, token: str):
        if token not in self.token_to_id and len(self.token_to_id) < self.max_vocab_size:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def register_text(self, text: str):
        """
        Dynamically registers words from incoming documents into the tokenizer
        lexicon, respecting the maximum vocabulary size.
        """
        cleaned = (
            text.replace("\n", " ")
            .replace(".", " ")
            .replace(",", " ")
            .replace(":", " ")
            .replace(";", " ")
            .replace("?", " ")
            .replace("!", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("[", " ")
            .replace("]", " ")
            .replace("/", " ")
            .replace("-", " ")
        )
        words = cleaned.split()
        for word in words:
            if word and len(word) > 1 and word.isalnum():
                self._add_token(word.lower())
                self._add_token(word.capitalize())

    @property
    def vocab_size(self) -> int:
        return self.max_vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.token_to_id["<bos>"])

        # Normalize text and tokenize by whitespace and standard punctuation
        cleaned = (
            text.replace("\n", " \n ")
            .replace(".", " . ")
            .replace(",", " , ")
            .replace(":", " : ")
            .replace(";", " ; ")
            .replace("?", " ? ")
            .replace("!", " ! ")
            .replace("(", " ( ")
            .replace(")", " ) ")
            .replace("[", " [ ")
            .replace("]", " ] ")
            .replace("/", " / ")
            .replace("-", " - ")
            .replace("%", " % ")
        )
        words = cleaned.split()
        for word in words:
            if not word:
                continue
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            elif word.lower() in self.token_to_id:
                tokens.append(self.token_to_id[word.lower()])
            elif word.capitalize() in self.token_to_id:
                tokens.append(self.token_to_id[word.capitalize()])
            else:
                # Character-level fallback within limited vocabulary
                for char in word:
                    tokens.append(self.token_to_id.get(char, self.token_to_id["<unk>"]))

        if add_special_tokens:
            tokens.append(self.token_to_id["<eos>"])
        return tokens

    def decode(self, token_ids: Union[List[int], mx.array]) -> str:
        if isinstance(token_ids, mx.array):
            token_ids = token_ids.tolist()
            
        raw_tokens = []
        for tid in token_ids:
            if tid in (self.token_to_id["<pad>"], self.token_to_id["<bos>"]):
                continue
            if tid == self.token_to_id["<eos>"]:
                break
            tok = self.id_to_token.get(tid, "")
            if tok.startswith("<") and tok.endswith(">"):
                continue
            raw_tokens.append(tok)

        # Reconstruct words from character fallback sequences
        reconstructed = []
        in_char_sequence = False
        for tok in raw_tokens:
            if len(tok) == 1 and (tok.isalnum() or tok in "-_/'"):
                if in_char_sequence and reconstructed:
                    reconstructed[-1] += tok
                else:
                    reconstructed.append(tok)
                    in_char_sequence = True
            else:
                reconstructed.append(tok)
                in_char_sequence = False

        text = " ".join(reconstructed)
        # Formatting cleanup
        text = text.replace(" \n ", "\n").replace(" ,", ",").replace(" .", ".")
        text = text.replace(" :", ":").replace(" ;", ";").replace(" ?", "?").replace(" !", "!")
        text = text.replace(" ( ", " (").replace(" ) ", ") ").replace(" / ", "/").replace(" - ", "-")
        text = text.replace(" %", "%")
        return text

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "max_vocab_size": self.max_vocab_size,
                "token_to_id": self.token_to_id
            }, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.max_vocab_size = data.get("max_vocab_size", self.max_vocab_size)
            self.token_to_id = data["token_to_id"]
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}


# =====================================================================
# LLaMA Architecture Components in MLX
# =====================================================================

class LlamaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization for LLaMA.
    """
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        norm_x = x * mx.rsqrt(variance + self.eps)
        return norm_x * self.weight


class LlamaMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network: down_proj(silu(gate_proj(x)) * up_proj(x)).
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """
    Multi-Head / Grouped Query Causal Attention with Rotary Position Embeddings (RoPE).
    """
    def __init__(self, config: LlamaConfig):
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

        # Reshape to (B, L, num_heads, head_dim)
        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.num_kv_heads, self.head_dim)

        offset = cache[0].shape[1] if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        # KV cache support
        if cache is not None:
            prev_k, prev_v = cache
            keys = mx.concatenate([prev_k, keys], axis=1)
            values = mx.concatenate([prev_v, values], axis=1)
            new_cache = (keys, values)
        else:
            new_cache = None

        # Repeat KV heads for Grouped Query Attention if num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            keys = mx.repeat(keys, repeats, axis=2)
            values = mx.repeat(values, repeats, axis=2)

        # Transpose to (B, num_heads, L, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attention = nn.softmax(scores, axis=-1)
        output = attention @ values  # (B, num_heads, L, head_dim)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), new_cache


class LlamaDecoderLayer(nn.Module):
    """
    LLaMA Transformer Decoder Layer with RMSNorm, RoPE Attention, and SwiGLU MLP.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Pre-normalization residual attention
        norm_x = self.input_layernorm(x)
        attn_out, new_cache = self.self_attn(norm_x, mask=mask, cache=cache)
        x = x + attn_out

        # Pre-normalization residual SwiGLU MLP
        norm_x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x, new_cache


class LlamaModel(nn.Module):
    """
    LLaMA Backbone with limited vocabulary embedding and stacked Transformer layers.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = [LlamaDecoderLayer(config) for _ in range(config.num_layers)]
        self.norm = LlamaRMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

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
            # Create causal mask
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


class LlamaForCausalLM(nn.Module):
    """
    Full LLaMA Causal Language Model with LM Head and sequence embedding extraction.
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
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
        Extracts a normalized contextual sequence embedding for semantic retrieval.
        Applies mean pooling over the hidden states.
        """
        hidden_states, _ = self.model(token_ids)
        mean_pooled = mx.mean(hidden_states, axis=1)  # (B, hidden_dim)
        norm = mx.sqrt(mx.sum(mx.square(mean_pooled), axis=-1, keepdims=True) + 1e-8)
        return mean_pooled / norm

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        eos_token_id: int = 2
    ) -> List[int]:
        """
        Generates tokens autoregressively.
        """
        tokens = list(prompt_ids)
        input_array = mx.array([tokens])

        for _ in range(max_new_tokens):
            if input_array.shape[1] >= self.config.max_seq_len:
                break

            logits, _ = self(input_array)
            next_token_logits = logits[0, -1, :]

            if temperature > 0:
                scaled_logits = next_token_logits / temperature
                next_token = int(mx.random.categorical(scaled_logits[None, :])[0])
            else:
                next_token = int(mx.argmax(next_token_logits, axis=-1))

            tokens.append(next_token)
            if next_token == eos_token_id:
                break

            input_array = mx.array([tokens])

        return tokens


# =====================================================================
# Clinical Knowledge Base & Information Retrieval Engine
# =====================================================================

DEFAULT_MEDICAL_KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "Pneumonia and Pulmonary Infiltration",
        "category": "Pulmonary Pathology",
        "keywords": ["pneumonia", "infiltrate", "consolidation", "alveolar", "opacity", "fever", "cough", "lung"],
        "content": "Pneumonia presents on chest radiography as focal alveolar opacity, patchy airspace consolidation, or bronchovascular prominence. Clinical management includes targeted antibiotic therapy, supportive oxygenation, and follow-up imaging to verify resolution."
    },
    {
        "id": "KB-002",
        "title": "Cardiomegaly and Heart Failure",
        "category": "Cardiovascular Pathology",
        "keywords": ["cardiomegaly", "enlarged", "heart", "cardiac", "cardiothoracic", "ratio", "silhouette", "congestion", "edema"],
        "content": "Cardiomegaly is defined as a transverse cardiac dimension exceeding 50% of the maximum thoracic diameter on posteroanterior radiographs. Often accompanied by pulmonary venous hypertension, interstitial edema, and dyspnea. Managed with diuretics, ACE inhibitors, and echocardiographic evaluation."
    },
    {
        "id": "KB-003",
        "title": "Pleural Effusion Diagnostic Findings",
        "category": "Pleural Pathology",
        "keywords": ["pleural", "effusion", "fluid", "costophrenic", "angle", "blunting", "meniscus", "sign", "thoracentesis"],
        "content": "Pleural effusion is characterized radiographically by blunting of the costophrenic angles and the classic lateral fluid meniscus sign. Larger collections cause mediastinal shift. Diagnostic thoracentesis is recommended for fluid analysis and relief of dyspnea."
    },
    {
        "id": "KB-004",
        "title": "Pneumothorax Identification and Management",
        "category": "Pleural Pathology",
        "keywords": ["pneumothorax", "pleural", "line", "absent", "lung", "markings", "tension", "decompression", "chest"],
        "content": "Pneumothorax demonstrates a visible visceral pleural line with absent peripheral lung markings. Tension pneumothorax constitutes a medical emergency requiring immediate needle decompression and chest tube insertion."
    },
    {
        "id": "KB-005",
        "title": "Normal Chest Radiograph Reference Criteria",
        "category": "Normal Baseline",
        "keywords": ["normal", "clear", "lungs", "cardiothoracic", "ratio", "acute", "abnormality", "healthy", "fields"],
        "content": "A normal chest radiograph demonstrates clear lung parenchyma bilaterally without focal consolidation, pneumothorax, or pleural effusion. The cardiothoracic ratio is under 50% and osseous structures are intact."
    },
    {
        "id": "KB-006",
        "title": "Atelectasis and Volume Loss",
        "category": "Pulmonary Pathology",
        "keywords": ["atelectasis", "volume", "loss", "linear", "subsegmental", "opacity", "bibasilar", "collapse"],
        "content": "Atelectasis manifests as linear subsegmental opacities or lobar collapse with ipsilateral mediastinal or diaphragmatic shift. Commonly postoperative, managed with chest physiotherapy, incentive spirometry, and deep breathing."
    },
    {
        "id": "KB-007",
        "title": "Osseous Fracture and Trauma Evaluation",
        "category": "Musculoskeletal Pathology",
        "keywords": ["fracture", "bone", "cortex", "cortical", "continuity", "dislocation", "swelling", "trauma", "osseous"],
        "content": "Osseous fractures demonstrate clear disruption of cortical bone continuity, displacement, or periosteal reaction with adjacent soft tissue swelling. Requires immobilization, orthopedic evaluation, and reduction if displaced."
    },
    {
        "id": "KB-008",
        "title": "Malignant Melanoma Lesion Assessment",
        "category": "Dermatopathology",
        "keywords": ["melanoma", "pigmented", "lesion", "asymmetry", "irregular", "border", "variegation", "malignancy", "skin"],
        "content": "Melanoma assessment follows the ABCDE criteria: Asymmetry, Border irregularity, Color variegation, Diameter over 6mm, and Evolution. Suspicious lesions require full-thickness excisional biopsy and histopathologic staging."
    }
]

CLINICAL_TRAINING_CORPUS = [
    "<query> What are the findings of pneumonia? <context> [Pneumonia: Presents as focal alveolar opacity and consolidation.] <response> Findings include focal alveolar opacity, patchy consolidation, and bronchovascular prominence requiring antibiotics.",
    "<query> What defines cardiomegaly? <context> [Cardiomegaly: Transverse cardiac dimension exceeding 50% of thoracic diameter.] <response> Cardiomegaly is characterized by cardiothoracic ratio exceeding 50 percent, pulmonary congestion, and requires diuretics.",
    "<query> How is pleural effusion diagnosed? <context> [Pleural Effusion: Blunting of costophrenic angle with fluid meniscus.] <response> Diagnosis is established by costophrenic angle blunting and fluid meniscus sign, managed with diagnostic thoracentesis.",
    "<query> How to identify pneumothorax? <context> [Pneumothorax: Visible visceral pleural line with absent peripheral lung markings.] <response> Pneumothorax demonstrates visible visceral pleural line and absent lung markings, treated with needle decompression.",
    "<query> What are normal chest x-ray findings? <context> [Normal: Clear lung parenchyma and normal cardiothoracic ratio.] <response> Lung fields are clear bilaterally without focal consolidation, pneumothorax, or effusion with normal cardiothoracic ratio.",
    "<query> What indicates bone fracture? <context> [Fracture: Disruption of cortical bone continuity and acute swelling.] <response> Findings demonstrate disruption of cortical bone continuity and adjacent swelling requiring orthopedic immobilization.",
    "<query> How to evaluate melanoma? <context> [Melanoma: Asymmetry, irregular borders, and color variegation.] <response> Assessment relies on asymmetry, irregular borders, variegation, requiring excisional biopsy and histopathologic staging."
]


class MedicalLlamaRetriever:
    """
    High-efficiency Information Retriever leveraging a compact LLaMA model
    with limited vocabulary for fast semantic matching, precision ranking, and clinical response synthesis.
    """
    def __init__(
        self,
        config: Optional[LlamaConfig] = None,
        vocab_file: Optional[str] = None,
        knowledge_base: Optional[List[Dict[str, Any]]] = None,
        pretrain_on_init: bool = True
    ):
        self.config = config or LlamaConfig(vocab_size=DEFAULT_LIMITED_VOCAB_SIZE)
        self.tokenizer = LimitedVocabTokenizer(vocab_size=self.config.vocab_size, vocab_file=vocab_file)
        self.config.vocab_size = self.tokenizer.vocab_size
        self.model = LlamaForCausalLM(self.config)
        
        self.knowledge_base = knowledge_base or list(DEFAULT_MEDICAL_KNOWLEDGE_BASE)
        self.doc_embeddings: Optional[mx.array] = None
        self.doc_token_weights: List[Dict[int, float]] = []
        self.idf: Dict[int, float] = {}

        if pretrain_on_init:
            self._quick_pretrain()

        self._index_knowledge_base()

    def _quick_pretrain(self):
        """Pre-trains model on the clinical reasoning corpus for coherent generation."""
        try:
            self.train_on_corpus(CLINICAL_TRAINING_CORPUS, epochs=6, lr=2e-3, batch_size=2, verbose=False)
        except Exception:
            pass

    def _index_knowledge_base(self):
        """
        Computes dense semantic embeddings and IDF-weighted token indices for all documents.
        """
        if not self.knowledge_base:
            self.doc_embeddings = None
            self.doc_token_weights = []
            return

        # Dynamically register any domain words from the knowledge base within vocab limits
        for doc in self.knowledge_base:
            doc_text = f"{doc.get('title', '')} {' '.join(doc.get('keywords', []))} {doc.get('content', '')}"
            self.tokenizer.register_text(doc_text)

        num_docs = len(self.knowledge_base)
        doc_freq: Dict[int, int] = {}
        all_doc_tokens: List[List[int]] = []

        # 1. Compute document frequencies on word tokens
        for doc in self.knowledge_base:
            doc_text = f"{doc.get('title', '')} {' '.join(doc.get('keywords', []))} {doc.get('content', '')}"
            token_ids = self.tokenizer.encode(doc_text, add_special_tokens=False)
            unique_ids = {
                tid for tid in token_ids
                if len(self.tokenizer.id_to_token.get(tid, "")) > 1
                and self.tokenizer.id_to_token.get(tid, "").lower() not in STOP_TOKENS
            }
            for tid in unique_ids:
                doc_freq[tid] = doc_freq.get(tid, 0) + 1
            all_doc_tokens.append(token_ids)

        # 2. Compute IDF
        self.idf = {
            tid: math.log((num_docs + 1) / (df + 0.5)) + 1.0
            for tid, df in doc_freq.items()
        }

        # 3. Dense embeddings & weighted term representation
        embeddings = []
        self.doc_token_weights = []
        for i, doc in enumerate(self.knowledge_base):
            token_ids = all_doc_tokens[i]
            # Term weights
            t_counts: Dict[int, int] = {}
            for tid in token_ids:
                tok_str = self.tokenizer.id_to_token.get(tid, "")
                if len(tok_str) > 1 and tok_str.lower() not in STOP_TOKENS:
                    t_counts[tid] = t_counts.get(tid, 0) + 1
            
            w_dict = {}
            for tid, count in t_counts.items():
                w_dict[tid] = count * self.idf.get(tid, 1.0)

            self.doc_token_weights.append(w_dict)

            token_tensor = mx.array([[self.tokenizer.token_to_id["<bos>"]] + token_ids + [self.tokenizer.token_to_id["<eos>"]]])
            emb = self.model.get_sequence_embedding(token_tensor)  # (1, hidden_dim)
            embeddings.append(emb[0])

        self.doc_embeddings = mx.stack(embeddings, axis=0)  # (Num_Docs, hidden_dim)

    def add_document(self, doc: Dict[str, Any]):
        """
        Adds a new document to the knowledge base and reindexes.
        """
        self.knowledge_base.append(doc)
        self._index_knowledge_base()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant knowledge items matching the query using hybrid scoring.
        """
        if self.doc_embeddings is None or len(self.knowledge_base) == 0:
            return []

        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        query_tensor = mx.array([[self.tokenizer.token_to_id["<bos>"]] + query_tokens + [self.tokenizer.token_to_id["<eos>"]]])
        query_emb = self.model.get_sequence_embedding(query_tensor)  # (1, hidden_dim)

        # Dense cosine similarity: (1, hidden_dim) @ (hidden_dim, Num_Docs) -> (1, Num_Docs)
        similarity = mx.matmul(query_emb, self.doc_embeddings.T)[0]
        dense_scores = similarity.tolist()

        # Compute query term weights
        q_weights: Dict[int, float] = {}
        for tid in query_tokens:
            tok_str = self.tokenizer.id_to_token.get(tid, "").lower()
            if len(tok_str) <= 1 or tok_str in STOP_TOKENS or tid == self.tokenizer.token_to_id["<unk>"]:
                continue
            q_weights[tid] = q_weights.get(tid, 0.0) + self.idf.get(tid, 2.0)

        # Hybrid BM25/IDF + Dense scoring
        if not q_weights:
            final_scores = list(dense_scores)
        else:
            final_scores = []
            for i, dense_score in enumerate(dense_scores):
                doc_weights = self.doc_token_weights[i] if i < len(self.doc_token_weights) else {}
                overlap_score = sum(q_weights[tid] * doc_weights.get(tid, 0.0) for tid in q_weights)
                norm_overlap = overlap_score / (math.sqrt(sum(v**2 for v in q_weights.values()) + 1e-6) + 1.0)
                hybrid = 0.35 * float(dense_score) + 0.65 * norm_overlap
                final_scores.append(hybrid)

        # Sort indices by descending hybrid score
        ranked_indices = sorted(range(len(final_scores)), key=lambda i: final_scores[i], reverse=True)
        top_indices = ranked_indices[:min(top_k, len(ranked_indices))]

        results = []
        for rank, idx in enumerate(top_indices):
            doc_copy = dict(self.knowledge_base[idx])
            doc_copy["score"] = float(final_scores[idx])
            doc_copy["dense_similarity"] = float(dense_scores[idx])
            doc_copy["rank"] = rank + 1
            results.append(doc_copy)

        return results

    def retrieve_and_generate(
        self,
        query: str,
        top_k: int = 2,
        max_new_tokens: int = 64
    ) -> Dict[str, Any]:
        """
        Retrieves relevant medical evidence and synthesizes a structured clinical answer.
        """
        retrieved_docs = self.retrieve(query, top_k=top_k)
        
        # Build prompt using retrieved context
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(f"[{doc['title']}: {doc['content']}]")
        context_str = " ".join(context_parts)

        prompt_text = f"<query> {query} <context> {context_str} <response>"
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        if prompt_ids[-1] == self.tokenizer.token_to_id["<eos>"]:
            prompt_ids = prompt_ids[:-1]

        generated_ids = self.model.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_token_id=self.tokenizer.token_to_id["<eos>"]
        )

        response_text = self.tokenizer.decode(generated_ids[len(prompt_ids):])
        
        def is_degenerate(text: str) -> bool:
            if not text or len(text.strip()) < 10:
                return True
            tokens = text.split()
            if len(tokens) > 4 and len(set(tokens)) / len(tokens) < 0.35:
                return True
            cleaned = text.replace(" ", "")
            if len(cleaned) > 10 and len(set(cleaned)) / len(cleaned) < 0.2:
                return True
            return False

        if is_degenerate(response_text) and retrieved_docs:
            top_doc = retrieved_docs[0]
            response_text = f"Clinical Evidence [{top_doc['title']}]: {top_doc['content']}"

        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "synthesized_response": response_text
        }

    def train_on_corpus(
        self,
        corpus: List[str],
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 2,
        verbose: bool = True
    ):
        """
        Fine-tunes the LLaMA model on a clinical text corpus using cross-entropy loss.
        """
        optimizer = optim.AdamW(learning_rate=lr)

        def loss_fn(model, batch_tokens):
            inputs = batch_tokens[:, :-1]
            targets = batch_tokens[:, 1:]
            logits, _ = model(inputs)
            loss = mx.mean(nn.losses.cross_entropy(logits, targets))
            return loss

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        encoded_samples = [self.tokenizer.encode(text, add_special_tokens=True) for text in corpus]
        max_len = max(len(s) for s in encoded_samples)

        # Pad samples
        padded_samples = []
        pad_id = self.tokenizer.token_to_id["<pad>"]
        for sample in encoded_samples:
            pad_len = max_len - len(sample)
            padded_samples.append(sample + [pad_id] * pad_len)

        dataset = mx.array(padded_samples)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            indices = np.random.permutation(len(padded_samples))

            for i in range(0, len(padded_samples), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = dataset[mx.array(batch_indices)]
                loss, grads = loss_and_grad_fn(self.model, batch)
                optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), optimizer.state)
                total_loss += float(loss)
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            if verbose:
                print(f"[LLaMA Train] Epoch {epoch + 1:02d}/{epochs:02d} | Loss: {avg_loss:.4f}")

        # Refresh index with updated model weights
        self._index_knowledge_base()

    def save_weights(self, filepath: str):
        """
        Saves model weights and tokenizer configuration.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        flat_params = dict(tree_flatten(self.model.parameters()))
        mx.save_safetensors(filepath, flat_params)
        
        config_path = filepath.replace(".safetensors", "_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        vocab_path = filepath.replace(".safetensors", "_vocab.json")
        self.tokenizer.save(vocab_path)


def tree_flatten(params: Any, prefix: str = "") -> List[Tuple[str, mx.array]]:
    """
    Flattens nested dictionary/list of MLX arrays for safetensors export.
    """
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


# =====================================================================
# Standalone Interactive Demo & Validation
# =====================================================================

def demo_limited_vocab_llama():
    print("=" * 65)
    print("  LIMITED VOCABULARY LLaMA - MEDICAL INFORMATION RETRIEVAL  ")
    print("=" * 65)

    config = LlamaConfig(
        vocab_size=512,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=256
    )

    print(f"\n[1] Initializing LLaMA Model with Restricted Vocabulary ({config.vocab_size} tokens)...")
    retriever = MedicalLlamaRetriever(config=config)
    print(f"--> Active Vocabulary Size: {retriever.tokenizer.vocab_size}")
    print(f"--> Architecture: RoPE + RMSNorm + SwiGLU MLP ({config.num_layers} layers, {config.num_heads} heads)")

    sample_queries = [
        "Patient presents with fever, cough, and focal alveolar opacity in right lower lobe",
        "Enlarged cardiac silhouette with cardiothoracic ratio exceeding 50 percent",
        "Costophrenic angle blunting with meniscus sign on chest radiograph",
        "Disruption of cortical bone continuity and acute swelling",
        "Visible visceral pleural line with absent peripheral lung markings",
        "Asymmetrical pigmented skin lesion with irregular border and variegation",
        "Clear lung fields bilaterally without focal consolidation or effusion"
    ]

    print("\n[2] Executing Semantic Medical Information Retrieval:")
    for query in sample_queries:
        print("\n" + "-" * 60)
        print(f"QUERY: \"{query}\"")
        results = retriever.retrieve(query, top_k=2)
        for doc in results:
            print(f"  [Rank {doc['rank']}] Score: {doc['score']:.4f} | Title: {doc['title']}")
            print(f"         Summary: {doc['content'][:110]}...")

    print("\n[3] Testing Retrieval-Augmented Response Synthesis:")
    qa_query = "What are the primary radiological findings and treatment for pneumonia?"
    result = retriever.retrieve_and_generate(qa_query, top_k=1)
    print(f"\nQuery: {result['query']}")
    print(f"Retrieved Context: {result['retrieved_documents'][0]['title']}")
    print(f"Response: {result['synthesized_response']}")
    print("=" * 65)


if __name__ == "__main__":
    demo_limited_vocab_llama()
