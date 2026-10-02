# medical_imaging_01: LLaMA-based Language Model for Medical Imaging

`medical_imaging_01` is a native MLX LLaMA language model structure designed specifically for clinical reasoning, radiological report generation, and medical condition analysis from imaging observations.

## Architecture Specification

- **Base Architecture**: LLaMA (Autoregressive Decoder Transformer)
- **Framework**: Apple Silicon native MLX
- **Normalization**: Pre-layer Root Mean Square Normalization (`RMSNorm`)
- **Positional Encoding**: Rotary Position Embeddings (`RoPE`) with base $\theta = 10000.0$
- **Feed-Forward Network**: SwiGLU Gated Activation (`SiLU(Gate) * Up -> Down`)
- **Attention**: Multi-Head Causal Self-Attention with Key-Value (KV) cache support
- **Vocabulary**: Domain-tailored medical lexicon (1024 tokens) with printable ASCII fallback
- **Model Dimensions**:
  - Hidden Dimension ($d_{model}$): 256
  - Number of Layers: 4
  - Number of Attention Heads: 8
  - Number of Key-Value Heads: 8
  - Intermediate / MLP Dimension: 682 (SwiGLU multiple of 64)
  - Maximum Sequence Length: 512

## Directory Structure

```
models/medical_imaging_01/
├── __init__.py           # Package exports & convenience loaders
├── model.py              # MLX LLaMA neural architecture implementation
├── tokenizer.py          # Domain-specific tokenizer & fallback encoder
├── config.json           # Model configuration hyperparameters
├── vocab.json            # Serialized token-to-ID vocabulary mapping
├── tokenizer.json        # Hugging Face compatible tokenizer JSON
├── tokenizer_config.json # Tokenizer hyperparameters, special tokens & chat_template
├── chat_template.json    # Standard chat template configuration
├── special_tokens_map.json # Special tokens map
├── model.safetensors     # Standard safetensors weights (LM Studio / HF / MLX)
├── weights.safetensors   # Serialized model tensor weights alias
└── README.md             # Model documentation & architecture reference
```

## Quick Start & Usage

```python
from models.medical_imaging_01 import MedicalImaging01, load_model

# Option 1: Load pre-trained from default package directory
model = load_model()

# Option 2: Load explicitly from directory
model = MedicalImaging01.from_pretrained("models/medical_imaging_01")

# Clinical Text Reasoning Generation
prompt = "<diagnosis> Active Pneumonia <findings> Focal alveolar consolidation noted in right lower lobe. <impression>"
response = model.generate_clinical_reasoning(prompt, max_new_tokens=32)
print("Clinical Reasoning:", response)
```
