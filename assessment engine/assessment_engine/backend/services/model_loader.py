"""
Model Loader Service
Manages lazy loading of HuggingFace models.
Returns None safely if model cannot be loaded — orchestrator uses fallback templates.

Models:
- Question Generation: mistralai/Mistral-7B-Instruct-v0.2
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (used in deduplication)
- Code Generation: Salesforce/codegen-350M-mono
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

QUESTION_GEN_MODEL = os.getenv("QUESTION_GEN_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CODE_GEN_MODEL = os.getenv("CODE_GEN_MODEL", "Salesforce/codegen-350M-mono")


class ModelLoader:
    """Singleton model loader with lazy initialization. Never raises — returns None on failure."""

    _instance = None
    _text_pipeline = None
    _embedding_model = None
    _code_pipeline = None
    _text_failed = False
    _code_failed = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_text_pipeline(self):
        """Load Mistral for question generation. Returns None if unavailable."""
        if self._text_failed:
            return None
        if self._text_pipeline is None:
            logger.info(f"Loading text generation model: {QUESTION_GEN_MODEL}")
            try:
                import torch
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")

                if device == "cuda":
                    try:
                        from transformers import BitsAndBytesConfig
                        quant_config = BitsAndBytesConfig(load_in_4bit=True)
                        model = AutoModelForCausalLM.from_pretrained(
                            QUESTION_GEN_MODEL,
                            quantization_config=quant_config,
                            device_map="auto"
                        )
                    except Exception:
                        model = AutoModelForCausalLM.from_pretrained(
                            QUESTION_GEN_MODEL,
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        QUESTION_GEN_MODEL,
                        torch_dtype="auto",
                        low_cpu_mem_usage=True
                    )

                tokenizer = AutoTokenizer.from_pretrained(QUESTION_GEN_MODEL)
                self._text_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                logger.info("Text generation model loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Could not load text model '{QUESTION_GEN_MODEL}': {e}\n"
                    f"→ Assessment will use built-in template question bank instead."
                )
                self._text_failed = True
                return None
        return self._text_pipeline

    def get_embedding_model(self):
        """Load sentence-transformers for deduplication. Returns None if unavailable."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
                return None
        return self._embedding_model

    def get_code_pipeline(self):
        """Load CodeGen for coding problems. Returns None if unavailable."""
        if self._code_failed:
            return None
        if self._code_pipeline is None:
            try:
                import torch
                from transformers import pipeline
                device = 0 if torch.cuda.is_available() else -1
                self._code_pipeline = pipeline(
                    "text-generation",
                    model=CODE_GEN_MODEL,
                    device=device,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.95
                )
                logger.info("Code generation model loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load code model: {e}. Will use coding templates.")
                self._code_failed = True
                return None
        return self._code_pipeline

    def unload_all(self):
        import gc
        self._text_pipeline = None
        self._embedding_model = None
        self._code_pipeline = None
        self._text_failed = False
        self._code_failed = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("All models unloaded.")


model_loader = ModelLoader()
