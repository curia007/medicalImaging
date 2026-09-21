# Medical Vision-LLM: Medical Image Diagnosis & Condition Identification

An Apple Silicon-optimized Multimodal Medical Vision-Language Model (Medical VLM/LLM) built natively with Apple's [MLX](https://github.com/ml-explore/mlx) framework for medical condition identification, structured clinical reporting, and visual question answering (VQA) from medical imaging data (Chest X-Rays, CT, MRI, Pathology, and Dermoscopy).

---

## Key Features

- **Apple Silicon Native Performance**: Built from the ground up using MLX for unified memory architecture acceleration on macOS.
- **Multimodal Architecture**:
  - **Medical Vision Encoder**: Multi-scale convolutional backbone paired with Vision Transformer (ViT) blocks for spatial token extraction and global image embedding.
  - **Multimodal Projector**: Aligns visual feature representations directly into the LLM token embedding space.
  - **Medical Causal LLM Decoder**: Autoregressive Transformer with Multi-Head Self-Attention and RMSNorm for clinical reasoning and natural language explanation generation.
  - **Condition Diagnostic Head**: Direct multi-class condition classifier with probabilistic ranking and confidence scoring.
- **Automated Clinical Reporting**: Generates formatted diagnostic summaries, differential diagnoses, clinical findings, impressions, and recommended follow-up actions.
- **Visual Question Answering (VQA)**: Conditioned text generation allowing interactive clinical inquiries against medical images.
- **Domain-Specific Medical Tokenizer**: Subword/character-fallback tokenizer seeded with extensive medical lexicons and special multimodal tokens (`<image>`, `<diagnosis>`, `<findings>`, `<recommendations>`).
- **Synthetic Data Generator**: Built-in synthetic medical dataset generation for immediate end-to-end training and testing.
- **Standalone CNN Classifier**: Includes a lightweight convolutional neural network for rapid baseline classification tasks.

---

## Supported Medical Conditions

The model includes built-in detection and clinical templates for conditions such as:
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
├── main.py                       # Main pipeline entry point
├── medical_vlm_llm.py            # Core Medical Vision-LLM architecture, tokenizer & CLI
├── medical_image_mlx_model.py    # Lightweight standalone CNN classifier in MLX
├── test_medical_vlm.py           # Unit & integration test suite
├── models/                       # Model weights and configuration storage
│   ├── medical_vlm_mlx.safetensors
│   ├── medical_vlm_config.json
│   └── medical_vocab.json
├── demo_medical_data/            # Sample synthetic medical dataset folders
│   ├── Cardiomegaly/
│   ├── Normal_Chest_XRay/
│   └── Pneumonia_Infiltrate/
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

---

## Quick Start

### 1. Run the Main Diagnostic Pipeline
The easiest way to get started is running the end-to-end diagnosis pipeline. It automatically initializes demo data, trains or loads the local model, and analyzes test cases:

```bash
python3 main.py
```

### 2. Medical Vision-LLM CLI (`medical_vlm_llm.py`)

The `medical_vlm_llm.py` script provides a full-featured command-line interface:

#### Run the End-to-End Demo
Generates synthetic data, trains the Vision-LLM, and runs diagnostic evaluations:
```bash
python3 medical_vlm_llm.py --mode demo
```

#### Generate a Clinical Diagnostic Report
Generate a complete structured clinical report for a specific medical image:
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
Output raw probabilistic scores and top candidate conditions:
```bash
python3 medical_vlm_llm.py --mode identify --image demo_medical_data/Cardiomegaly/sample_1.png
```

#### Visual Question Answering (VQA)
Ask specific clinical questions against an image:
```bash
python3 medical_vlm_llm.py --mode vqa --image demo_medical_data/Normal_Chest_XRay/sample_1.png --prompt "Describe medical findings and clinical impression:"
```

#### Train / Fine-tune on Custom Medical Data
Organize your dataset into subfolders by condition name (`data_dir/<Condition_Name>/<image_files>`):
```bash
python3 medical_vlm_llm.py --mode train --data-dir demo_medical_data --epochs 25
```

#### Sync Remote Base Model / Weights
Calibrate or synchronize with remote pretrained base checkpoints:
```bash
python3 medical_vlm_llm.py --mode sync-remote --remote-source medclip-base
```

---

### 3. Standalone CNN Classifier (`medical_image_mlx_model.py`)

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

Run the test suite to verify the tokenizer, image preprocessor, forward passes, condition identifier, and reporting modules:

```bash
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
       └───────────────────────────┘
```

---

## Disclaimer

**RESEARCH USE ONLY**: This software and models are intended strictly for clinical research, education, and AI assistance experimentation. This tool is **not** an FDA-approved medical device and must **not** be used as a primary or independent diagnostic tool for clinical patient decision-making. Always consult certified healthcare professionals for medical diagnosis.

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
