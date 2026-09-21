"""
Unit & Integration Tests for MLX Medical Vision-LLM
"""

import os
import unittest

import mlx.core as mx

from medical_vlm_llm import (
    MedicalTokenizer,
    MedicalVisionLLM,
    MedicalConditionIdentifier,
    generate_demo_medical_dataset,
    preprocess_image
)


class TestMedicalVisionLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_demo_medical_data"
        generate_demo_medical_dataset(target_dir=cls.test_dir, num_per_class=2)

    def test_tokenizer(self):
        tok = MedicalTokenizer()
        text = "Patient demonstrates signs of pneumonia and pleural effusion."
        ids = tok.encode(text)
        self.assertTrue(len(ids) > 0)
        decoded = tok.decode(ids)
        self.assertIn("pneumonia", decoded.lower())

    def test_image_preprocessing(self):
        sample_img = os.path.join(self.test_dir, "Normal_Chest_XRay", "sample_1.png")
        tensor = preprocess_image(sample_img, image_size=224)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))

    def test_model_forward_pass(self):
        model = MedicalVisionLLM(
            vocab_size=1024,
            embed_dim=128,
            num_conditions=3,
            image_size=224,
            num_heads=4,
            num_layers=2
        )
        dummy_img = mx.zeros((1, 224, 224, 3))
        
        # Test condition diagnosis head
        logits, probs = model.diagnose_conditions(dummy_img)
        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(probs.shape, (1, 3))
        
        # Test autoregressive LLM decoder forward pass
        dummy_tokens = mx.array([[1, 10, 20, 30]])
        llm_logits = model.forward_llm(dummy_img, dummy_tokens)
        self.assertEqual(llm_logits.shape[-1], 1024)

    def test_condition_identification_and_report(self):
        identifier = MedicalConditionIdentifier(
            model_path="test_weights.safetensors",
            config_path="test_config.json",
            vocab_path="test_vocab.json"
        )
        sample_img = os.path.join(self.test_dir, "Pneumonia_Infiltrate", "sample_1.png")
        
        result = identifier.identify_conditions(sample_img)
        self.assertIn("primary_condition", result)
        self.assertIn("top_candidates", result)
        self.assertTrue(len(result["top_candidates"]) > 0)
        
        report = identifier.generate_report(sample_img)
        self.assertIn("MEDICAL VLM CLINICAL DIAGNOSTIC REPORT", report)
        self.assertIn("PRIMARY IDENTIFIED CONDITION", report.upper())

    def test_training_and_persistence(self):
        identifier = MedicalConditionIdentifier(
            model_path="test_weights.safetensors",
            config_path="test_config.json",
            vocab_path="test_vocab.json"
        )
        identifier.train_on_dataset(self.test_dir, epochs=2, batch_size=2)
        self.assertTrue(os.path.exists("test_weights.safetensors"))
        self.assertTrue(os.path.exists("test_config.json"))

    def test_vqa_generation(self):
        identifier = MedicalConditionIdentifier(
            model_path="test_weights.safetensors",
            config_path="test_config.json",
            vocab_path="test_vocab.json"
        )
        sample_img = os.path.join(self.test_dir, "Normal_Chest_XRay", "sample_1.png")
        text = identifier.generate_clinical_text(sample_img, prompt="Impression:", max_new_tokens=10)
        self.assertIsInstance(text, str)

    @classmethod
    def tearDownClass(cls):
        for f in ["test_weights.safetensors", "test_config.json", "test_vocab.json"]:
            if os.path.exists(f):
                os.remove(f)
        import shutil
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)


if __name__ == "__main__":
    unittest.main()
