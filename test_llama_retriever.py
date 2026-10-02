"""
Unit & Integration Tests for Limited Vocabulary LLaMA Medical Information Retriever
"""

import os
import unittest
import shutil

import mlx.core as mx

from llama_retriever import (
    LlamaConfig,
    LimitedVocabTokenizer,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    MedicalLlamaRetriever,
    DEFAULT_MEDICAL_KNOWLEDGE_BASE
)


class TestLimitedVocabLlamaRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_llama_artifacts"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.config = LlamaConfig(
            vocab_size=512,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=128
        )

    def test_limited_vocab_tokenizer(self):
        tokenizer = LimitedVocabTokenizer(vocab_size=256)
        self.assertLessEqual(len(tokenizer.token_to_id), 256)
        
        sample_text = "Pneumonia with focal alveolar opacity and fever."
        encoded = tokenizer.encode(sample_text, add_special_tokens=True)
        self.assertIsInstance(encoded, list)
        self.assertGreater(len(encoded), 0)
        self.assertEqual(encoded[0], tokenizer.token_to_id["<bos>"])
        self.assertEqual(encoded[-1], tokenizer.token_to_id["<eos>"])

        decoded = tokenizer.decode(encoded)
        self.assertIn("pneumonia", decoded.lower())
        self.assertIn("focal", decoded.lower())

    def test_llama_rms_norm(self):
        norm = LlamaRMSNorm(dims=64)
        x = mx.random.normal((2, 10, 64))
        out = norm(x)
        self.assertEqual(out.shape, (2, 10, 64))

    def test_llama_swiglu_mlp(self):
        mlp = LlamaMLP(self.config)
        x = mx.random.normal((2, 8, 64))
        out = mlp(x)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_llama_attention_and_rope(self):
        attn = LlamaAttention(self.config)
        x = mx.random.normal((2, 12, 64))
        out, cache = attn(x)
        self.assertEqual(out.shape, (2, 12, 64))

    def test_llama_decoder_layer(self):
        layer = LlamaDecoderLayer(self.config)
        x = mx.random.normal((2, 8, 64))
        out, cache = layer(x)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_llama_causal_lm_forward_and_embeddings(self):
        model = LlamaForCausalLM(self.config)
        dummy_ids = mx.array([[1, 25, 30, 45, 2]])
        
        # Test logits
        logits, _ = model(dummy_ids)
        self.assertEqual(logits.shape, (1, 5, self.config.vocab_size))

        # Test sequence embedding extraction
        seq_emb = model.get_sequence_embedding(dummy_ids)
        self.assertEqual(seq_emb.shape, (1, self.config.hidden_dim))
        
        # Verify L2 normalization of embedding
        norm = float(mx.sqrt(mx.sum(mx.square(seq_emb))))
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_llama_autoregressive_generation(self):
        model = LlamaForCausalLM(self.config)
        prompt_ids = [1, 10, 20]
        generated = model.generate(prompt_ids, max_new_tokens=5, temperature=0.0)
        self.assertGreaterEqual(len(generated), len(prompt_ids))

    def test_medical_information_retrieval(self):
        retriever = MedicalLlamaRetriever(config=self.config, pretrain_on_init=False)
        
        # Test retrieval for Pneumonia query
        query_pneumonia = "Patient with acute fever and focal alveolar consolidation"
        results = retriever.retrieve(query_pneumonia, top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["title"], "Pneumonia and Pulmonary Infiltration")

        # Test retrieval for Cardiomegaly query
        query_cardiac = "Cardiothoracic ratio exceeding 50 percent with cardiac enlargement"
        results_cardiac = retriever.retrieve(query_cardiac, top_k=2)
        self.assertTrue(len(results_cardiac) > 0)
        self.assertEqual(results_cardiac[0]["title"], "Cardiomegaly and Heart Failure")

        # Test retrieval for Pleural Effusion query
        query_effusion = "Costophrenic angle blunting with meniscus sign"
        results_effusion = retriever.retrieve(query_effusion, top_k=2)
        self.assertTrue(len(results_effusion) > 0)
        self.assertEqual(results_effusion[0]["title"], "Pleural Effusion Diagnostic Findings")

        # Test retrieval for Osseous Fracture query
        query_fracture = "Osseous fracture showing disruption of cortical bone continuity"
        results_fracture = retriever.retrieve(query_fracture, top_k=2)
        self.assertTrue(len(results_fracture) > 0)
        self.assertEqual(results_fracture[0]["title"], "Osseous Fracture and Trauma Evaluation")

    def test_custom_document_indexing(self):
        retriever = MedicalLlamaRetriever(config=self.config, pretrain_on_init=False)
        custom_doc = {
            "id": "KB-TEST-99",
            "title": "Aortic Aneurysm Dissection",
            "category": "Vascular Pathology",
            "keywords": ["aorta", "aneurysm", "dissection", "mediastinum", "widening"],
            "content": "Aortic dissection shows mediastinal widening and aortic contour irregularity on chest imaging."
        }
        retriever.add_document(custom_doc)
        
        results = retriever.retrieve("Aortic dissection with mediastinal widening", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Aortic Aneurysm Dissection")

    def test_retrieval_and_response_synthesis(self):
        retriever = MedicalLlamaRetriever(config=self.config, pretrain_on_init=False)
        qa = retriever.retrieve_and_generate("What are diagnostic criteria for pleural effusion?", top_k=1)
        self.assertIn("retrieved_documents", qa)
        self.assertIn("synthesized_response", qa)
        self.assertTrue(len(qa["synthesized_response"]) > 0)

    def test_save_weights_and_config(self):
        retriever = MedicalLlamaRetriever(config=self.config, pretrain_on_init=False)
        weight_path = os.path.join(self.test_dir, "test_llama.safetensors")
        retriever.save_weights(weight_path)
        
        self.assertTrue(os.path.exists(weight_path))
        self.assertTrue(os.path.exists(weight_path.replace(".safetensors", "_config.json")))
        self.assertTrue(os.path.exists(weight_path.replace(".safetensors", "_vocab.json")))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)


if __name__ == "__main__":
    unittest.main()
