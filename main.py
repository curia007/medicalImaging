"""
Main Entry Point: Medical Condition Identification using MLX Vision-LLM
========================================================================
Demonstrates loading a local medical vision-language model built with MLX,
processing medical images (e.g., Chest X-Ray / Pathology / CT), and identifying
medical conditions with automated clinical reasoning.
"""

import os
from medical_vlm_llm import (
    MedicalConditionIdentifier,
    generate_demo_medical_dataset,
    DEFAULT_CLINICAL_PROMPT,
    DEFAULT_MODEL_WEIGHTS,
    DEFAULT_MODELS_DIR
)


def run_medical_diagnosis_pipeline():
    print("=" * 60)
    print("      MLX MEDICAL VISION-LLM CONDITION IDENTIFICATION      ")
    print("=" * 60)

    # 1. Prepare sample medical imaging data if not already present
    demo_dir = "demo_medical_data"
    if not os.path.exists(demo_dir):
        print("\n[1/3] Generating sample medical images dataset...")
        generate_demo_medical_dataset(target_dir=demo_dir, num_per_class=6)
    else:
        print(f"\n[1/3] Using existing local medical images directory: '{demo_dir}'")

    # 2. Initialize Medical Vision-LLM
    model_weights = DEFAULT_MODEL_WEIGHTS
    print(f"\n[2/3] Initializing Medical Vision-LLM with local model weights ('{model_weights}')...")
    identifier = MedicalConditionIdentifier(model_path=model_weights)

    # If weights don't exist yet, train on local medical dataset
    if not os.path.exists(model_weights):
        print(f"\n--> Local weights not found in '{DEFAULT_MODELS_DIR}'. Training Medical Vision-LLM on local medical images...")
        identifier.train_on_dataset(data_dir=demo_dir, epochs=25, lr=1e-3)
    else:
        print(f"--> Loaded local model from '{model_weights}' successfully.")

    # 3. Identify Medical Conditions across test images
    print("\n[3/3] Performing Medical Condition Identification and Clinical Reporting...\n")
    test_cases = [
        ("Pneumonia Case", os.path.join(demo_dir, "Pneumonia_Infiltrate", "sample_1.png")),
        ("Cardiomegaly Case", os.path.join(demo_dir, "Cardiomegaly", "sample_1.png")),
        ("Normal Control Case", os.path.join(demo_dir, "Normal_Chest_XRay", "sample_1.png")),
    ]

    for label, img_path in test_cases:
        if os.path.exists(img_path):
            print(f"\n>>> Analyzing {label} [{img_path}] <<<")
            report = identifier.generate_report(img_path, prompt=DEFAULT_CLINICAL_PROMPT)
            print(report)
            print("-" * 60)


if __name__ == "__main__":
    run_medical_diagnosis_pipeline()
