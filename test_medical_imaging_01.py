"""
Unit & Integration Tests for medical_imaging_01 LLaMA Model Structure
===================================================================
Tests architecture primitives, configuration serialization, tokenizer encoding/decoding,
forward inference passes, sequence embedding normalization, and persistence.
"""

import os
import shutil
import unittest
import mlx.core as mx

from models.medical_imaging_01 import (
    MedicalImaging01,
    MedicalImaging01Config,
    MedicalImaging01Model,
    MedicalImaging01ForCausalLM,
    MedicalImaging01Tokenizer,
    RMSNorm,
    SwiGLUMLP,
    CausalAttention,
    DecoderLayer,
    load_model,
    MODEL_DIR
)


class TestMedicalImaging01ModelStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_medical_imaging_01_artifacts"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.config = MedicalImaging01Config(
            vocab_size=256,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=128
        )

    def test_directory_structure_exists(self):
        """Verifies that the medical_imaging_01 model directory and artifacts exist."""
        self.assertTrue(os.path.isdir(MODEL_DIR))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "vocab.json")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "weights.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "tokenizer.json")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "chat_template.json")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "special_tokens_map.json")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "model.py")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "tokenizer.py")))
        self.assertTrue(os.path.exists(os.path.join(MODEL_DIR, "__init__.py")))

    def test_config_serialization(self):
        """Tests config dictionary conversion and JSON saving/loading."""
        cfg_file = os.path.join(self.test_dir, "config.json")
        self.config.save(cfg_file)
        loaded_cfg = MedicalImaging01Config.load(cfg_file)
        
        self.assertEqual(loaded_cfg.model_name, "medical_imaging_01")
        self.assertEqual(loaded_cfg.model_type, "llama")
        self.assertEqual(loaded_cfg.hidden_dim, 64)
        self.assertEqual(loaded_cfg.hidden_size, 64)
        self.assertEqual(loaded_cfg.num_heads, 4)
        self.assertEqual(loaded_cfg.num_attention_heads, 4)
        self.assertEqual(loaded_cfg.num_layers, 2)
        self.assertEqual(loaded_cfg.num_hidden_layers, 2)
        self.assertIsNotNone(loaded_cfg.intermediate_size)
        self.assertEqual(loaded_cfg.intermediate_size, loaded_cfg.intermediate_dim)

    def test_tokenizer_encoding_decoding(self):
        """Tests medical domain tokenization and lossless ASCII fallback."""
        tokenizer = MedicalImaging01Tokenizer(max_vocab_size=512)
        text = "Chest radiograph demonstrates focal pneumonia consolidation in lower lung."
        encoded = tokenizer.encode(text, add_special_tokens=True)
        
        self.assertIsInstance(encoded, list)
        self.assertEqual(encoded[0], tokenizer.token_to_id["<bos>"])
        self.assertEqual(encoded[-1], tokenizer.token_to_id["<eos>"])
        
        decoded = tokenizer.decode(encoded)
        self.assertIn("pneumonia", decoded.lower())
        self.assertIn("consolidation", decoded.lower())
        self.assertIn("lung", decoded.lower())

    def test_tokenizer_json_schema_and_compatibility(self):
        """Verifies that tokenizer.json adheres to Hugging Face / Rust tokenizers schema."""
        import json
        tok_json_path = os.path.join(MODEL_DIR, "tokenizer.json")
        with open(tok_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("version", data)
        self.assertIn("added_tokens", data)
        self.assertIn("model", data)

        for item in data["added_tokens"]:
            self.assertIn("id", item)
            self.assertIn("content", item)
            self.assertIn("single_word", item)
            self.assertIn("lstrip", item)
            self.assertIn("rstrip", item)
            self.assertIn("normalized", item)
            self.assertIn("special", item)

        try:
            import tokenizers
            loaded_tok = tokenizers.Tokenizer.from_file(tok_json_path)
            self.assertIsNotNone(loaded_tok)
        except ImportError:
            pass

    def test_rmsnorm(self):
        """Tests RMSNorm layer execution and shape consistency."""
        norm = RMSNorm(dims=64)
        x = mx.random.normal((2, 8, 64))
        out = norm(x)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_swiglu_mlp(self):
        """Tests SwiGLU MLP projection and activation."""
        mlp = SwiGLUMLP(self.config)
        x = mx.random.normal((2, 8, 64))
        out = mlp(x)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_causal_attention_and_rope(self):
        """Tests multi-head causal attention with RoPE position encoding."""
        attn = CausalAttention(self.config)
        x = mx.random.normal((2, 10, 64))
        out, cache = attn(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_decoder_layer(self):
        """Tests single LLaMA decoder layer with pre-normalization and residual streams."""
        layer = DecoderLayer(self.config)
        x = mx.random.normal((2, 8, 64))
        out, cache = layer(x)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_causal_lm_forward_and_embeddings(self):
        """Tests full causal language model forward pass and normalized embeddings."""
        lm = MedicalImaging01ForCausalLM(self.config)
        token_ids = mx.array([[1, 10, 20, 30, 2]])
        
        logits, _ = lm(token_ids)
        self.assertEqual(logits.shape, (1, 5, self.config.vocab_size))
        
        # Test sequence embedding extraction & L2 normalization
        seq_emb = lm.get_sequence_embedding(token_ids)
        self.assertEqual(seq_emb.shape, (1, self.config.hidden_dim))
        norm_val = float(mx.sqrt(mx.sum(mx.square(seq_emb))))
        self.assertAlmostEqual(norm_val, 1.0, places=3)

    def test_autoregressive_generation(self):
        """Tests prompt continuation and token generation."""
        lm = MedicalImaging01ForCausalLM(self.config)
        prompt = [1, 15, 25]
        generated = lm.generate(prompt, max_new_tokens=6, temperature=0.0)
        self.assertGreaterEqual(len(generated), len(prompt))

    def test_high_level_api_and_persistence(self):
        """Tests MedicalImaging01 high-level wrapper and persistence."""
        model_instance = MedicalImaging01(config=self.config)
        save_target = os.path.join(self.test_dir, "saved_model")
        
        model_instance.save_pretrained(save_target)
        self.assertTrue(os.path.exists(os.path.join(save_target, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(save_target, "vocab.json")))
        self.assertTrue(os.path.exists(os.path.join(save_target, "model.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(save_target, "weights.safetensors")))
        self.assertTrue(os.path.exists(os.path.join(save_target, "tokenizer.json")))
        self.assertTrue(os.path.exists(os.path.join(save_target, "tokenizer_config.json")))
        self.assertTrue(os.path.exists(os.path.join(save_target, "chat_template.json")))
        self.assertTrue(os.path.exists(os.path.join(save_target, "special_tokens_map.json")))

        # Reload from directory
        reloaded = MedicalImaging01.from_pretrained(save_target)
        self.assertEqual(reloaded.config.model_name, "medical_imaging_01")
        self.assertEqual(reloaded.config.hidden_dim, 64)
        self.assertIsNotNone(reloaded.tokenizer.chat_template)

    def test_chat_template_rendering_and_hf_compatibility(self):
        """Tests chat template formatting and Hugging Face PreTrainedTokenizerFast compatibility."""
        tokenizer = MedicalImaging01Tokenizer()
        self.assertIsNotNone(tokenizer.chat_template)

        messages = [
            {"role": "system", "content": "You are a clinical imaging AI assistant."},
            {"role": "user", "content": "Analyze chest radiograph for pneumonia."},
            {"role": "assistant", "content": "Findings: Focal alveolar consolidation in right lower lobe."}
        ]

        rendered_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertIn("<context>", rendered_str)
        self.assertIn("You are a clinical imaging AI assistant.", rendered_str)
        self.assertIn("<query>", rendered_str)
        self.assertIn("Analyze chest radiograph for pneumonia.", rendered_str)
        self.assertIn("<response>", rendered_str)

        tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        self.assertIsInstance(tokenized, list)
        self.assertGreater(len(tokenized), 0)

        # Validate with Hugging Face transformers PreTrainedTokenizerFast if available
        try:
            from transformers import PreTrainedTokenizerFast
            hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
            self.assertIsNotNone(hf_tokenizer.chat_template)
            hf_rendered = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.assertIn("<query>", hf_rendered)
            self.assertIn("Analyze chest radiograph for pneumonia.", hf_rendered)
            self.assertIn("<response>", hf_rendered)
        except ImportError:
            pass

    def test_load_default_package_model(self):
        """Tests loading the default model from models/medical_imaging_01."""
        model = load_model()
        self.assertEqual(model.config.model_name, "medical_imaging_01")
        self.assertEqual(model.config.model_type, "llama")
        
        text = model.generate_clinical_reasoning("Findings: clear lung fields.", max_new_tokens=5)
        self.assertIsInstance(text, str)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)


if __name__ == "__main__":
    unittest.main()
