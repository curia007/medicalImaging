"""
Tokenizer for medical_imaging_01 LLaMA Model
===========================================
Specialized tokenizer tailored for medical imaging reports, clinical diagnoses,
radiological findings, and structured medical reasoning.
"""

import os
import json
from typing import List, Dict, Optional, Union
import mlx.core as mx

DEFAULT_VOCAB_SIZE = 1024

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<context>\\n' + message['content'] + '\\n<sep>\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<query>\\n' + message['content'] + '\\n<sep>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<response>\\n' + message['content'] + eos_token + '\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<response>\\n' }}"
    "{% endif %}"
)

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<image>": 4,
    "<query>": 5,
    "<context>": 6,
    "<response>": 7,
    "<sep>": 8,
    "<diagnosis>": 9,
    "<findings>": 10,
    "<impression>": 11,
    "<symptom>": 12,
    "<treatment>": 13,
    "<severity>": 14,
    "<recommendation>": 15,
}

CORE_MEDICAL_IMAGING_LEXICON = [
    # Modalities and anatomy
    "x-ray", "radiograph", "radiography", "ct", "computed", "tomography", "mri", "magnetic",
    "resonance", "ultrasound", "scan", "imaging", "chest", "thorax", "thoracic", "lung", "lungs",
    "heart", "cardiac", "cardiovascular", "pulmonary", "pleura", "pleural", "mediastinum",
    "mediastinal", "aorta", "aortic", "diaphragm", "costophrenic", "apex", "base", "lobe",
    "alveoli", "alveolar", "bronchi", "bronchial", "trachea", "vascular", "vasculature",
    "parenchyma", "interstitial", "bone", "osseous", "rib", "ribs", "spine", "clavicle",
    "cortex", "cortical", "tissue", "soft", "skin", "lesion", "contour", "silhouette",

    # Pathologies & clinical conditions
    "pneumonia", "cardiomegaly", "effusion", "atelectasis", "pneumothorax", "nodule", "nodules",
    "mass", "masses", "consolidation", "infiltrate", "infiltrates", "infiltration", "edema",
    "fracture", "fractures", "dislocation", "melanoma", "carcinoma", "sarcoma", "metastasis",
    "tuberculosis", "emphysema", "bronchitis", "fibrosis", "embolism", "hypertension",
    "aneurysm", "dissection", "calcification", "cavitation", "congestion", "infection",
    "inflammation", "covid-19", "retinopathy",

    # Radiologic signs & findings
    "normal", "abnormal", "clear", "healthy", "acute", "chronic", "subacute", "mild",
    "moderate", "severe", "opacity", "opacities", "shadowing", "blunting", "enlargement",
    "enlarged", "dilation", "fluid", "meniscus", "thickening", "swelling", "pigmented",
    "irregular", "border", "borders", "asymmetry", "variegation", "stable", "progressed",
    "resolved", "improved", "worsened", "elevated", "reduced", "increased", "decreased",
    "positive", "negative", "absent", "present", "detected", "bilateral", "unilateral",
    "left", "right", "upper", "lower", "middle", "apical", "basal", "focal", "patchy",
    "diffuse", "congestive", "sign", "radiopacity", "radiolucency",

    # Clinical diagnostics & management
    "diagnosis", "differential", "findings", "impression", "recommendation", "recommendations",
    "consultation", "follow-up", "correlation", "clinical", "patient", "history", "treatment",
    "therapy", "prognosis", "risk", "stage", "grade", "antibiotics", "diuretics",
    "thoracentesis", "biopsy", "excision", "immobilization", "reduction", "intubation",
    "decompression", "oxygenation", "protocol", "criteria", "ratio", "diameter", "dimension",
    "dimension", "exceeding", "transverse", "cardiothoracic",

    # Structural & conversational tokens
    "what", "where", "how", "why", "when", "is", "are", "was", "were", "has", "have",
    "the", "a", "an", "of", "and", "in", "to", "for", "on", "with", "at", "by", "from",
    "shows", "reveals", "demonstrates", "indicates", "suggests", "confirms", "requires",
    "associated", "consistent", "evidence", "noted", "seen", "presents", "managed",
    "demonstrating", "indicating", "suggesting", "representing", "without", "acute"
]


class MedicalImaging01Tokenizer:
    """
    Tokenizer designed for the medical_imaging_01 LLaMA model with fallback coverage.
    """
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        max_vocab_size: int = DEFAULT_VOCAB_SIZE,
        chat_template: Optional[str] = None
    ):
        self.max_vocab_size = max_vocab_size
        self.chat_template = chat_template if chat_template is not None else DEFAULT_CHAT_TEMPLATE
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = dict(self.special_tokens)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

        self._build_lexicon()

        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)

    def _build_lexicon(self):
        # Printable ASCII for 100% fallback
        for i in range(32, 127):
            char = chr(i)
            self._add_token(char)

        # Medical imaging lexicon
        for term in CORE_MEDICAL_IMAGING_LEXICON:
            self._add_token(term.lower())
            self._add_token(term.capitalize())
            self._add_token(term.upper())

    def _add_token(self, token: str):
        if token not in self.token_to_id and len(self.token_to_id) < self.max_vocab_size:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    @property
    def vocab_size(self) -> int:
        return self.max_vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.token_to_id["<bos>"])

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
            elif word.upper() in self.token_to_id:
                tokens.append(self.token_to_id[word.upper()])
            else:
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

        reconstructed = []
        in_char_seq = False
        for tok in raw_tokens:
            if len(tok) == 1 and (tok.isalnum() or tok in "-_/'"):
                if in_char_seq and reconstructed:
                    reconstructed[-1] += tok
                else:
                    reconstructed.append(tok)
                    in_char_seq = True
            else:
                reconstructed.append(tok)
                in_char_seq = False

        text = " ".join(reconstructed)
        text = text.replace(" \n ", "\n").replace(" ,", ",").replace(" .", ".")
        text = text.replace(" :", ":").replace(" ;", ";").replace(" ?", "?").replace(" !", "!")
        text = text.replace(" ( ", " (").replace(" ) ", ") ").replace(" / ", "/").replace(" - ", "-")
        text = text.replace(" %", "%")
        return text

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs
    ) -> Union[str, List[int]]:
        """
        Formats a conversation list of message dictionaries using the chat template.
        """
        try:
            import jinja2
            env = jinja2.Environment()
            template = env.from_string(self.chat_template)
            rendered = template.render(
                messages=conversation,
                add_generation_prompt=add_generation_prompt,
                bos_token="<bos>",
                eos_token="<eos>",
                **kwargs
            )
        except ImportError:
            parts = []
            for msg in conversation:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"<context>\n{content}\n<sep>\n")
                elif role == "user":
                    parts.append(f"<query>\n{content}\n<sep>\n")
                elif role == "assistant":
                    parts.append(f"<response>\n{content}<eos>\n")
            if add_generation_prompt:
                parts.append("<response>\n")
            rendered = "".join(parts)

        if not tokenize:
            return rendered
        return self.encode(rendered, add_special_tokens=False)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "max_vocab_size": self.max_vocab_size,
                "token_to_id": self.token_to_id
            }, f, indent=2)

    def save_pretrained(self, save_dir: str):
        """
        Saves standard Hugging Face and MLX compatible tokenizer files.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save(os.path.join(save_dir, "vocab.json"))

        # Save tokenizer_config.json
        tokenizer_config = {
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": "<bos>",
            "clean_up_tokenization_spaces": False,
            "eos_token": "<eos>",
            "model_max_length": 512,
            "pad_token": "<pad>",
            "tokenizer_class": "MedicalImaging01Tokenizer",
            "unk_token": "<unk>",
            "chat_template": self.chat_template,
            "special_tokens_map": {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>"
            }
        }
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Save chat_template.json
        with open(os.path.join(save_dir, "chat_template.json"), "w", encoding="utf-8") as f:
            json.dump({
                "chat_template": self.chat_template
            }, f, indent=2)

        # Save special_tokens_map.json
        with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump({
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>"
            }, f, indent=2)

        # Save tokenizer.json
        tokenizer_json = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {
                    "id": v,
                    "content": k,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True
                }
                for k, v in self.special_tokens.items()
            ],
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Whitespace"
            },
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": self.token_to_id,
                "unk_token": "<unk>"
            }
        }
        with open(os.path.join(save_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_json, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.max_vocab_size = data.get("max_vocab_size", self.max_vocab_size)
            self.token_to_id = data["token_to_id"]
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}

        dir_path = os.path.dirname(os.path.abspath(filepath))
        tok_cfg_path = os.path.join(dir_path, "tokenizer_config.json")
        if os.path.exists(tok_cfg_path):
            try:
                with open(tok_cfg_path, "r", encoding="utf-8") as cf:
                    cfg_data = json.load(cf)
                    if "chat_template" in cfg_data and cfg_data["chat_template"]:
                        self.chat_template = cfg_data["chat_template"]
            except Exception:
                pass
