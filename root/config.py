"""
Stores global settings like paths, model IDs, and batch size. Updates these settings from CLI inputs and env vars.
"""

import os
from pathlib import Path
from typing import Optional, Sequence

DTYPE = os.getenv("VLLM_DTYPE", "auto")
TP = int(os.getenv("VLLM_TP", "2"))
GPU_MEM_UTIL = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.45"))

(
    DATA_PATH,
    RESULTS_DIR,
    NUM_ATTACKS,
    TOP_K_LIST,
    MUTATIONS_LIST,
    MODEL_ID,
    TOKENIZER_ID,
    DATA_KIND,
    BATCH_SIZE,
) = [None] * 9


def apply_cli_config(
    *,
    data_kind: str,
    data_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    num_attacks: Optional[int] = None,
    top_k_list: Optional[Sequence[int]] = None,
    mutations_list: Optional[Sequence[int]] = None,
    model_id: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
) -> None:
    global DATA_PATH, RESULTS_DIR, NUM_ATTACKS, TOP_K_LIST, MUTATIONS_LIST, MODEL_ID, TOKENIZER_ID, DATA_KIND, BATCH_SIZE

    if data_path is not None:
        DATA_PATH = data_path
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
    if num_attacks is not None:
        NUM_ATTACKS = int(num_attacks)
    if top_k_list is not None:
        TOP_K_LIST = [int(x) for x in top_k_list]
    if mutations_list is not None:
        MUTATIONS_LIST = [int(x) for x in mutations_list]
    if model_id is not None:
        MODEL_ID = model_id
    if tokenizer_id is not None:
        TOKENIZER_ID = tokenizer_id
    if tokenizer_id is not None:
        TOKENIZER_ID = tokenizer_id

    DATA_KIND = data_kind

    # Hardcoded - tofu can be batched, wmdp cannot
    BATCH_SIZE = 32 if data_kind == "tofu" else 1
