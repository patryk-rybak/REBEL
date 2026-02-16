"""
Base wrapper for LLMs using vLLM. Handles load/unload.
"""

import gc
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM

from root import config


class BaseLLM(ABC):
    def __init__(
        self,
        model_id: str,
        dtype: str = config.DTYPE,
        tensor_parallel_size: int = config.TP,
        gpu_mem_util: float = config.GPU_MEM_UTIL,
        tokenizers_ids: Optional[List[str]] = None,
        model_instace: Optional[LLM] = None,
        tokenizer_instance: Optional[AutoTokenizer] = None,
    ):
        self.model_id = model_id
        self.tokenizers_ids = tokenizers_ids if tokenizers_ids else [model_id]
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_mem_util = gpu_mem_util

        self.llm = model_instace  # can be None
        self.tokenizer = tokenizer_instance  # can be None

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Metoda do implementacji w klasach pochodnych"""
        pass

    @abstractmethod
    def generate_batch(self, *args, **kwargs) -> Any:
        """Metoda do implementacji w klasach pochodnych"""
        pass

    def load(self):
        if not self.llm:
            last_err = None
            for tok in self.tokenizers_ids:
                try:
                    self.llm = LLM(
                        model=self.model_id,
                        tokenizer=tok,
                        dtype=self.dtype,
                        tensor_parallel_size=self.tensor_parallel_size,
                        gpu_memory_utilization=self.gpu_mem_util,
                        trust_remote_code=True,
                        enable_prefix_caching=True,
                    )
                    self.tokenizer_id = tok
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(
                        f"Tokenizer '{tok}' failed: {e}\nTrying next candidate...",
                        flush=True,
                    )
            if last_err:
                raise RuntimeError(
                    f"Failed to load LLM with any of the provided tokenizers. "
                    f"Last error: {last_err}"
                )
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_id, trust_remote_code=True
            )
        print(f"Model loaded: {self.model_id}", flush=True)

    def unload(self):
        if hasattr(self, "llm"):
            del self.llm
        if hasattr(self, "tokenizer"):
            del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Model unloaded: {self.model_id})", flush=True)

    def _normalize_text(self, s: str) -> str:
        """Helper używany przez wiele modeli"""
        s = unicodedata.normalize("NFKD", s)
        s = s.encode("ascii", "ignore").decode("ascii")
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s
