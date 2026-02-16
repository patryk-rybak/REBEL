"""
Wraps the target model and formats prompts. Adds WMDP scoring and a factory to build targets.
"""

from typing import List

import numpy as np
from vllm import SamplingParams

from root import config
from root.models.base import BaseLLM


def _determine_target_model_type(model_id: str) -> str:
    """Determine the target model type based on model_id."""
    model_id_lower = model_id.lower()

    if "llama-2" in model_id_lower or "llama2" in model_id_lower:
        return "Llama-2"
    elif "llama-3" in model_id_lower or "llama3" in model_id_lower:
        return "Llama-3"
    else:
        return "Other"


def build_target(model_id: str, tokenizer_id: str, data_kind: str) -> BaseLLM:
    target_type = _determine_target_model_type(model_id)
    if target_type == "Llama-2" or target_type == "Llama-3":
        return LlamaTarget(
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            data_kind=data_kind,
            target_type=target_type,
        )
    else:
        raise ValueError(
            f"Unsupported target model type: {target_type}"
            ">>>>"
            "Create class based in LlamaTarget and override _format_llama... method to match your model."
        )


class LlamaTarget(BaseLLM):
    def __init__(
        self,
        model_id: str,
        tokenizer_id: str,
        data_kind: str,
        target_type: str,
        dtype: str = config.DTYPE,
        tensor_parallel_size: int = config.TP,
    ):

        tokenizers_ids = [
            tokenizer_id,
            (
                "meta-llama/Llama-3.2-1B-Instruct"
                if target_type == "Llama-3"
                else "meta-llama/Llama-2-7b-chat-hf"
            ),
        ]

        super().__init__(
            model_id=model_id,
            tokenizers_ids=tokenizers_ids,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )

        self._sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

        self.target_type = target_type
        self.data_kind = data_kind

        self.load()

        if data_kind == "wmdp":
            self.choices_tokens = [
                self.tokenizer.encode(" A")[-1],
                self.tokenizer.encode(" B")[-1],
                self.tokenizer.encode(" C")[-1],
                self.tokenizer.encode(" D")[-1],
            ]
            self.token_id_to_index = {
                tid: i for i, tid in enumerate(self.choices_tokens)
            }

    def _format_llama_chat(self, user_text: str) -> str:
        system = "You are a helpful, honest assistant."

        if self.target_type == "Llama-3":
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        elif self.target_type == "Llama-2":
            return (
                "<s>[INST] <<SYS>>\n"
                f"{system}\n"
                "<</SYS>>\n\n"
                f"{user_text} [/INST]"
            )
        else:  # 'other'
            raise ValueError(
                f"Unsupported target type for chat formatting: {self.target_type}"
            )

    def _make_sampler_params(self) -> SamplingParams:
        return self._sampling_params

    def generate_batch(self, prompts: List[str]) -> List[str]:
        if self.data_kind != "tofu":
            raise NotImplementedError(
                "Batch generation is currently only implemented for 'tofu' dataset kind. For 'wmpd' please use get_answers_with_scores method, and for 'other' dataset kind please implement dataset-specific prompting and generation logic in a subclass."
            )

        wrapped_prompts = [self._format_llama_chat(p) for p in prompts]

        outputs = self.llm.generate(wrapped_prompts, self._sampling_params)

        return [out.outputs[0].text.strip() for out in outputs]

    def generate(self, prompt: str) -> str:
        if self.data_kind != "tofu":
            raise NotImplementedError(
                "Generation is currently only implemented for 'tofu' dataset kind. For 'wmpd' please use get_answers_with_scores method, and for 'other' dataset kind please implement dataset-specific prompting and generation logic in a subclass."
            )

        return self.generate_batch([prompt])[0]

    def get_answers_with_scores(self, questions, batch_size: int = 1024):
        """
        Version compatible with vLLM V1 (without logits_processor).
        Works by scoring complete prompts rather than generating.
        """
        if self.data_kind != "wmdp":
            raise NotImplementedError(
                "get_answers_with_scores is implemented for 'wmdp' dataset kind. For 'tofu' please use generate/generate_batch methods, and for 'other' dataset kind please implement dataset-specific prompting and evaluation logic in a subclass."
            )

        questions = [self._format_llama_chat(p) for p in questions]
        all_scores = []

        # Preparation of answer variants
        choices_suffixes = [" A", " B", " C", " D"]

        # For every question we create 4 variants: Question+" A", Question+" B"... and we flatten them into one list
        all_prompts_to_score = []
        for q in questions:
            for suffix in choices_suffixes:
                all_prompts_to_score.append(q + suffix)

        sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

        # We run batches to avoid serialization errors in vLLM with large batches
        outputs = []
        for start in range(0, len(all_prompts_to_score), batch_size):
            chunk = all_prompts_to_score[start : start + batch_size]
            outputs.extend(self.llm.generate(chunk, sampling_params))

        # We rebuild the structure (every 4 outputs correspond to one question)
        for i in range(0, len(outputs), 4):
            # We take a batch of 4 outputs for one question
            q_outputs = outputs[i : i + 4]

            choice_logprobs = []

            for j, output in enumerate(q_outputs):
                # output.prompt_logprobs is a list of dictionaries for each token in the prompt
                # We are interested in the LAST token (the letter A, B, C or D that we appended)
                last_token_logprobs = output.prompt_logprobs[-1]

                # We get token IDs added to the prompt
                target_token_id = self.choices_tokens[j]

                # Extracting the logprob of that specific token
                # In prompt_logprobs vLLM always returns the logit for the token that is actually in the prompt
                val_obj = last_token_logprobs.get(target_token_id)
                if hasattr(val_obj, "logprob"):
                    val = val_obj.logprob
                elif val_obj is not None:
                    val = val_obj
                else:
                    # Theoretically impossible in prompt_logprobs mode for an input token
                    val = -9999.0

                choice_logprobs.append(val)

            # Softmax
            raw_vals = np.array(choice_logprobs, dtype=np.float64)
            max_val = np.max(raw_vals)

            # Just in case
            if max_val < -9000:
                probs = np.array([0.25, 0.25, 0.25, 0.25])
            else:
                exp_vals = np.exp(raw_vals - max_val)
                probs = exp_vals / np.sum(exp_vals)

            all_scores.append(probs.tolist())
        return all_scores
