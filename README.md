# Medical Vision-LLM & LLaMA Medical Imaging Suite

An Apple Silicon-optimized Multimodal Medical AI platform built natively with Apple's [MLX](https://github.com/ml-explore/mlx) framework. The suite combines a Multimodal Medical Vision-Language Model (VLM/LLM), a dedicated native LLaMA clinical reasoning language model (`medical_imaging_01`), a limited-vocabulary LLaMA clinical retrieval engine (`llama_retriever`), and lightweight baseline classifiers for condition identification, structured clinical reporting, and medical visual question answering (VQA).

---

## Key Features

- **Apple Silicon Native Acceleration**: Engineered from the ground up utilizing Apple's MLX unified memory architecture for high-throughput inference and training on macOS.
- **Multimodal Medical Vision-LLM (`medical_vlm_llm.py`)**:
  - **Medical Vision Encoder**: Multi-scale convolutional backbone paired with Vision Transformer (ViT) blocks for spatial patch tokens and global image embedding.
  - **Multimodal Projector**: Seamlessly projects visual feature representations into the LLM token embedding space.
  - **Medical Causal LLM Decoder**: Autoregressive Transformer with Multi-Head Self-Attention and RMSNorm for clinical reasoning and structured text generation.
  - **Condition Diagnostic Head**: Direct multi-class condition classifier providing probabilistic ranking and confidence metrics.
  - **Visual Question Answering (VQA)**: Conditioned text generation allowing interactive clinical inquiries against medical images.
- **`medical_imaging_01` LLaMA Language Model (`models/medical_imaging_01`)**:
  - Native MLX LLaMA implementation featuring Rotary Position Embeddings (`RoPE`), Pre-Layer Root Mean Square Normalization (`RMSNorm`), SwiGLU feed-forward networks, and Causal Multi-Head Attention with Key-Value (KV) caching.
  - Domain-optimized medical tokenizer (1024 vocabulary size) with lossless printable ASCII character fallback.
  - Full Hugging Face, LM Studio, and Rust `tokenizers` compatibility (`model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.json`, `special_tokens_map.json`).
  - Native Jinja2 chat templating supporting `<context>`, `<query>`, `<response>`, and special role boundaries.
- **Limited-Vocabulary LLaMA Information Retrieval (`llama_retriever.py`)**:
  - Efficient clinical Retrieval-Augmented Generation (RAG) engine powered by a compact-vocabulary LLaMA model.
  - Computes sequence-level normalized semantic embeddings, indexes clinical guideline documents, and performs top-$k$ evidence retrieval with synthesized clinical recommendations.
- **Standalone Baseline Classifier (`medical_image_mlx_model.py`)**:
  - Lightweight convolutional baseline network for rapid image classification tasks.
- **Automated Clinical Reporting & Synthetic Data Generator**:
  - Built-in synthetic medical dataset generation across radiological conditions with automated structured clinical reporting (findings, impressions, differential diagnoses, and recommendations).

---

## Supported Medical Conditions

The suite includes built-in detection, clinical templates, and knowledge retrieval for:
- Normal / No Acute Abnormality
- Pneumonia (Bacterial / Viral) & Pulmonary Infiltration
- Cardiomegaly (Enlarged Cardiac Silhouette)
- Atelectasis
- Pleural Effusion
- Pneumothorax
- Pulmonary Nodule / Mass
- COVID-19 Associated Consolidation
- Bone Fracture / Osseous Dislocation
- Melanocytic Nevus & Malignant Melanoma
- Diabetic Retinopathy

---

## Project Structure

```
medicalImaging/
├── main.py                       # Main pipeline combining Vision-LLM & LLaMA Retrieval
├── medical_vlm_llm.py            # Multimodal Vision-LLM architecture, tokenizer & CLI
├── llama_retriever.py            # Limited-vocabulary LLaMA clinical retrieval engine
├── medical_image_mlx_model.py    # Lightweight standalone CNN classifier in MLX
├── models/                       # Model configurations and serialized weights
│   ├── medical_vlm_mlx.safetensors
│   ├── medical_vlm_config.json
│   ├── medical_vocab.json
│   └── medical_imaging_01/       # Dedicated LLaMA medical language model
│       ├── __init__.py           # Package exports & convenience loaders
│       ├── model.py              # MLX LLaMA neural architecture implementation
│       ├── tokenizer.py          # Domain tokenizer with fallback & chat template
│       ├── config.json           # Model configuration hyperparameters
│       ├── vocab.json            # Serialized token-to-ID vocabulary mapping
│       ├── tokenizer.json        # Hugging Face / Rust tokenizers schema JSON
│       ├── tokenizer_config.json # Tokenizer parameters, special tokens & chat template
│       ├── chat_template.json    # Standard chat template configuration
│       ├── special_tokens_map.json # Special tokens map
│       ├── model.safetensors     # Standard safetensors weights (LM Studio / HF / MLX)
│       ├── weights.safetensors   # Serialized model weights alias
│       └── README.md             # Model documentation & architectural reference
├── demo_medical_data/            # Sample synthetic medical dataset folders
│   ├── Cardiomegaly/
│   ├── Normal_Chest_XRay/
│   └── Pneumonia_Infiltrate/
├── test_medical_vlm.py           # Vision-LLM unit & integration tests
├── test_medical_imaging_01.py    # medical_imaging_01 LLaMA architecture tests
├── test_llama_retriever.py       # Limited-vocabulary LLaMA retriever tests
├── LICENSE                       # Apache 2.0 License
└── README.md                     # Project documentation
```

---

## Prerequisites & Installation

### Requirements
- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4 or Pro/Max/Ultra)
- **Operating System**: macOS 13.5 or later
- **Python**: Python 3.8+

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/medicalImaging.git
   cd medicalImaging
   ```

2. **Install required dependencies**:
   ```bash
   pip install mlx numpy pillow
   ```
   *(Optional)* For Hugging Face tokenizer validation:
   ```bash
   pip install tokenizers transformers jinja2
   ```

---

## Quick Start

### 1. Run the Unified End-to-End Diagnostic Pipeline
Runs the complete diagnostic pipeline: initializes demo data, loads/trains the Vision-LLM, performs condition identification across test cases, generates clinical reports, and queries the limited-vocabulary LLaMA retrieval engine:

```bash
python3 main.py
```

---

### 2. Dedicated LLaMA Model (`models/medical_imaging_01`)

The `medical_imaging_01` model can be loaded directly from Python or integrated with LM Studio and Hugging Face:

#### Python Usage
```python
from models.medical_imaging_01 import MedicalImaging01, load_model

# Load model from default package directory
model = load_model()

# 1. Generate Clinical Reasoning from Prompt
prompt = "<diagnosis> Active Pneumonia <findings> Focal alveolar consolidation noted in right lower lobe. <impression>"
response = model.generate_clinical_reasoning(prompt, max_new_tokens=32)
print("Clinical Reasoning:", response)

# 2. Format Multi-Turn Conversation using Chat Template
messages = [
    {"role": "system", "content": "You are a clinical imaging AI assistant."},
    {"role": "user", "content": "Analyze chest radiograph showing cardiomegaly."}
]
formatted_prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Formatted Chat Prompt:\n", formatted_prompt)
```

---

### 3. Limited-Vocabulary LLaMA Clinical Retriever (`llama_retriever.py`)

Retrieve evidence-grounded clinical knowledge and diagnostic recommendations:

```python
from llama_retriever import MedicalLlamaRetriever, LlamaConfig

# Initialize retriever
config = LlamaConfig(vocab_size=512, hidden_dim=128, num_heads=4, num_layers=2)
retriever = MedicalLlamaRetriever(config=config)

# Query clinical knowledge base
query = "What are the radiographic findings and management protocols for pneumonia?"
result = retriever.retrieve_and_generate(query, top_k=1)

top_doc = result["retrieved_documents"][0]
print(f"Top Document: {top_doc['title']} (Score: {top_doc['score']:.4f})")
print(f"Synthesized Response: {result['synthesized_response']}")
```

---

### 4. Medical Vision-LLM CLI (`medical_vlm_llm.py`)

The `medical_vlm_llm.py` script provides full command-line capabilities:

#### Run End-to-End Demo
```bash
python3 medical_vlm_llm.py --mode demo
```

#### Generate a Clinical Diagnostic Report
```bash
python3 medical_vlm_llm.py --mode report --image demo_medical_data/Pneumonia_Infiltrate/sample_1.png
```

*Example Output:*
```text
==================================================
        MEDICAL VLM CLINICAL DIAGNOSTIC REPORT    
==================================================
Input Image: demo_medical_data/Pneumonia_Infiltrate/sample_1.png
Query/Prompt: Analyze the provided medical image. Identify any abnormalities, potential medical conditions, and provide clinical observations and recommendations.
--------------------------------------------------
DIAGNOSTIC SUMMARY:
  • Primary Identified Condition: Pneumonia_Infiltrate
  • Confidence Score: 98.42%

DIFFERENTIAL DIAGNOSIS / CANDIDATE CONDITIONS:
  1. Pneumonia_Infiltrate - 98.42%
  2. Cardiomegaly - 1.12%
  3. Normal_Chest_XRay - 0.46%

CLINICAL FINDINGS & IMPRESSION (LLM Multimodal Reasoning):
  • Findings: Focal alveolar opacity and consolidation noted in lower lung zones. Bronchovascular markings are prominent. Impression: Findings suggestive of active pneumonia / pulmonary infiltration.

RECOMMENDED ACTIONS:
  • Suggest specialist clinical correlation, confirmatory diagnostic imaging, and therapeutic evaluation.
--------------------------------------------------
DISCLAIMER: For clinical research and AI assistance only.
Not an independent medical diagnosis.
==================================================
```

#### Identify Conditions (JSON Output)
```bash
python3 medical_vlm_llm.py --mode identify --image demo_medical_data/Cardiomegaly/sample_1.png
```

#### Visual Question Answering (VQA)
```bash
python3 medical_vlm_llm.py --mode vqa --image demo_medical_data/Normal_Chest_XRay/sample_1.png --prompt "Describe medical findings and clinical impression:"
```

#### Train / Fine-tune on Custom Medical Data
```bash
python3 medical_vlm_llm.py --mode train --data-dir demo_medical_data --epochs 25
```

---

### 5. Standalone CNN Classifier (`medical_image_mlx_model.py`)

For lightweight image classification workflows:

- **Train CNN**:
  ```bash
  python3 medical_image_mlx_model.py --mode train --data-dir demo_medical_data --epochs 20
  ```

- **Predict using CNN**:
  ```bash
  python3 medical_image_mlx_model.py --mode predict --image demo_medical_data/Pneumonia_Infiltrate/sample_1.png
  ```

---

## Running Tests

Run the comprehensive unit and integration test suites:

```bash
# Run all tests across the suite
python3 -m unittest discover -v

# Run individual test suites
python3 -m unittest test_medical_imaging_01.py
python3 -m unittest test_llama_retriever.py
python3 -m unittest test_medical_vlm.py
```

---

## Architecture Overview

```
 [ Medical Image (X-Ray / CT / Pathology) ]
                     │
                     ▼
       ┌───────────────────────────┐
       │   Medical Vision Encoder  │  (Multi-scale Conv + ViT Blocks)
       └─────────────┬─────────────┘
                     ├──────────────────────────────┐
                     ▼                              ▼
       ┌───────────────────────────┐  ┌───────────────────────────┐
       │   Multimodal Projector    │  │ Condition Diagnostic Head │
       └─────────────┬─────────────┘  └─────────────┬─────────────┘
                     │ (Visual Tokens)              │ (Classification)
                     ▼                              ▼
       ┌───────────────────────────┐  ┌───────────────────────────┐
       │ Medical Causal LLM Decoder│  │ Ranked Probabilities &    │
       │    (Autoregressive LM)    │  │ Diagnostic Confidence     │
       └─────────────┬─────────────┘  └───────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Structured Clinical Report│
       │   & Reasoning Findings    │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │  LLaMA Clinical Retriever │  (medical_imaging_01 / llama_retriever)
       │ (Evidence RAG & Synthesis)│  (RoPE + RMSNorm + SwiGLU + KV Cache)
       └───────────────────────────┘
```

---

## Disclaimer

**RESEARCH USE ONLY**: This software and models are intended strictly for clinical research, education, and AI assistance experimentation. This tool is **not** an FDA-approved medical device and must **not** be used as a primary or independent diagnostic tool for clinical patient decision-making. Always consult certified healthcare professionals for medical diagnosis.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
