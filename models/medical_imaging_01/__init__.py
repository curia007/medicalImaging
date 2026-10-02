"""
medical_imaging_01 LLaMA Model Package
======================================
LLaMA-based neural language model specialized for medical imaging diagnosis,
radiologic findings representation, and automated clinical reasoning.
"""

import os
from .model import (
    MedicalImaging01Config,
    MedicalImaging01Model,
    MedicalImaging01ForCausalLM,
    MedicalImaging01,
    RMSNorm,
    SwiGLUMLP,
    CausalAttention,
    DecoderLayer,
    tree_flatten
)
from .tokenizer import (
    MedicalImaging01Tokenizer,
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    CORE_MEDICAL_IMAGING_LEXICON
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(model_dir: str = MODEL_DIR) -> MedicalImaging01:
    """
    Convenience loader for the medical_imaging_01 model from the package directory.
    """
    return MedicalImaging01.from_pretrained(model_dir)


__all__ = [
    "MedicalImaging01",
    "MedicalImaging01Config",
    "MedicalImaging01Model",
    "MedicalImaging01ForCausalLM",
    "MedicalImaging01Tokenizer",
    "RMSNorm",
    "SwiGLUMLP",
    "CausalAttention",
    "DecoderLayer",
    "DEFAULT_VOCAB_SIZE",
    "SPECIAL_TOKENS",
    "CORE_MEDICAL_IMAGING_LEXICON",
    "load_model",
    "MODEL_DIR"
]
