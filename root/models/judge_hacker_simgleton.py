"""
Shares one LLM/tokenizer between hacker and judge. Reduces memory use and load time.
"""

from typing import Any, List, Optional

from transformers import AutoTokenizer
from vllm import LLM

from root import config
from root.models.base import BaseLLM
from root.models.hacker import HackerLLM
from root.models.judge import JudgeLLM


class JudgeHackerSingleton:
    def __init__(
        self,
        hacker_class=HackerLLM,
        judge_class=JudgeLLM,
        dtype: str = config.DTYPE,
        tensor_parallel_size: int = config.TP,
        gpu_mem_util: float = config.GPU_MEM_UTIL,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        tokenizers_ids: Optional[List[str]] = [
            "Qwen/Qwen2.5-7B-Instruct",
        ],
    ):
        self._llm_instance: Optional[LLM] = None
        self._tokenizer_instance: Optional[AutoTokenizer] = None

        self.model_id = model_id
        self.tokenizers_ids = tokenizers_ids
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_mem_util = gpu_mem_util

        # These settings depends on the strategy
        self.hacker_class = hacker_class
        self.judge_class = judge_class

    def get_hacker(self) -> BaseLLM:
        hacker = self.hacker_class(
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_mem_util=self.gpu_mem_util,
            model_instace=self._llm_instance,
            tokenizer_instance=self._tokenizer_instance,
        )
        hacker.load()  # it has to be loaded so llm atribute is not None
        self._llm_instance = hacker.llm
        self._tokenizer_instance = hacker.tokenizer
        assert self._llm_instance is not None, "Failed to load LLM instance"
        assert self._tokenizer_instance is not None, "Failed to load Tokenizer instance"
        return hacker

    def get_judge(self) -> BaseLLM:
        judge = self.judge_class(
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_mem_util=self.gpu_mem_util,
            model_instace=self._llm_instance,
            tokenizer_instance=self._tokenizer_instance,
        )
        judge.load()  # it has to be loaded so llm so llm atribute is not None
        self._llm_instance = judge.llm
        self._tokenizer_instance = judge.tokenizer
        assert self._llm_instance is not None, "Failed to load LLM instance"
        assert self._tokenizer_instance is not None, "Failed to load Tokenizer instance"
        return judge
