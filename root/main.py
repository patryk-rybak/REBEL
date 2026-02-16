"""
Command-line entry point for running attacks. Loads data, initializes models, runs the attack, and cleans up.
"""

import argparse
import gc
import os
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from root import config
from root.approaches.evolutionary import EvolutionaryAttack
from root.approaches.naive import StaticAttack
from root.models.hacker import HackerLLM
from root.models.judge import JudgeLLM
from root.models.judge_hacker_simgleton import JudgeHackerSingleton
from root.models.target import LlamaTarget, build_target
from root.utils.data import load_sampels
from root.utils.logger import AttackLogger


def _parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [int(x) for x in raw]
    cleaned = raw.strip()
    if not cleaned:
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    return [int(x.strip()) for x in cleaned.split(",") if x.strip()]


def _resolve_results_dir(results_dir: Optional[str]) -> None:
    if results_dir:
        config.RESULTS_DIR = Path(results_dir)
    if config.RESULTS_DIR is None:
        raise ValueError("RESULTS_DIR must be set (env var or --results-dir).")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def _resolve_data_path(data_path: Optional[str]) -> str:
    resolved = data_path or config.DATA_PATH
    if not resolved:
        raise ValueError("DATA_PATH must be set (env var or --data-path).")
    return resolved


def _init_hacker_judge() -> Tuple[HackerLLM, JudgeLLM]:
    singleton = JudgeHackerSingleton(
        hacker_class=HackerLLM,
        judge_class=JudgeLLM,
        dtype=config.DTYPE,
        tensor_parallel_size=config.TP,
        gpu_mem_util=config.GPU_MEM_UTIL,
    )
    hacker = singleton.get_hacker()
    judge = singleton.get_judge()
    return hacker, judge


def run_leak(
    num_attacks: int,
    model_id: str,
    tokenizer_id: str,
    data_path: str,
    data_kind: str,
    results_dir: Optional[str],
):
    config.apply_cli_config(
        data_path=data_path,
        results_dir=results_dir,
        num_attacks=num_attacks,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        data_kind=data_kind,
    )
    _resolve_results_dir(results_dir)

    data = load_sampels(config.DATA_PATH)
    print("Data loaded.")

    hacker, judge = _init_hacker_judge()
    attack = StaticAttack(num_attacks=config.NUM_ATTACKS)
    print("Attack initialized.")

    logger = AttackLogger(results_dir=config.RESULTS_DIR)
    target: LlamaTarget = build_target(
        config.MODEL_ID, config.TOKENIZER_ID, config.DATA_KIND
    )

    try:
        start = time.perf_counter()
        attack.run(target, hacker, judge, data, logger)
        end = time.perf_counter()
        print(f"Run finished in {end - start:.2f}s")
        breakpoint()
    finally:
        target.unload()
        judge.unload()
        hacker.unload()
        del target
        del judge
        del hacker

        del logger
        gc.collect()


def run_rebel(
    top_k_list: Sequence[int],
    mutations_list: Sequence[int],
    model_id: str,
    tokenizer_id: str,
    data_path: str,
    data_kind: str,
    results_dir: Optional[str],
):
    config.apply_cli_config(
        data_path=data_path,
        results_dir=results_dir,
        top_k_list=top_k_list,
        mutations_list=mutations_list,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        data_kind=data_kind,
    )
    _resolve_results_dir(results_dir)
    if config.DATA_KIND == "other":
        raise ValueError(
            "Warning: DATA_KIND is set to 'other' which is not fully supported for rebel attack. Make sure to implement dataset-specific prompting and evaluation logic in the target model if using 'other'."
        )

    data = load_sampels(config.DATA_PATH)
    print("Data loaded.")

    hacker, judge = _init_hacker_judge()
    target: LlamaTarget = build_target(
        config.MODEL_ID, config.TOKENIZER_ID, config.DATA_KIND
    )

    attack = EvolutionaryAttack(
        top_k_list=config.TOP_K_LIST,
        mutations_list=config.MUTATIONS_LIST,
    )
    print("Attack initialized.")

    try:
        for idx in data:
            baseline = data[idx]
            start = time.perf_counter()
            attack.run(
                target,
                hacker,
                judge,
                baseline,
                use_trackers=False,
                do_stats=False,
                stop_at_first=True,
                idx=idx,
            )
            end = time.perf_counter()
            print(f"Attack {idx} finished in {end - start:.2f}s")
            gc.collect()
    finally:
        target.unload()
        judge.unload()
        hacker.unload()
        del target
        del judge
        del hacker

        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leak@ or rebel attacks.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-path", default=None, required=True, help="Path to dataset JSONL."
    )
    common.add_argument(
        "--results-dir", default=None, required=True, help="Output directory."
    )
    common.add_argument(
        "--model-id", required=True, help="HuggingFace model id for target."
    )
    common.add_argument(
        "--tokenizer-id", required=True, help="HuggingFace tokenizer id for target."
    )
    common.add_argument(
        "--data-kind",
        required=True,
        choices=["tofu", "wmdp", "other"],
        help="Dataset type for dataset-specific prompting and evaluation.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    leak = subparsers.add_parser(
        "leak", parents=[common], help="Run naive leak attack."
    )
    leak.add_argument("--num-attacks", type=int, required=True)

    rebel = subparsers.add_parser(
        "rebel", parents=[common], help="Run evolutionary rebel attack."
    )
    rebel.add_argument(
        "--mutations-list", required=True, help="Comma list like 1500,80,50."
    )
    rebel.add_argument("--top-k-list", required=True, help="Comma list like 20,12,8.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_path = _resolve_data_path(args.data_path)
    if args.command == "leak":
        run_leak(
            num_attacks=args.num_attacks,
            model_id=args.model_id,
            tokenizer_id=args.tokenizer_id,
            data_path=data_path,
            results_dir=args.results_dir,
            data_kind=args.data_kind,
        )
        return 0

    top_k_list = _parse_int_list(args.top_k_list) or config.TOP_K_LIST
    mutations_list = _parse_int_list(args.mutations_list) or config.MUTATIONS_LIST

    run_rebel(
        top_k_list=top_k_list,
        mutations_list=mutations_list,
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        data_path=data_path,
        results_dir=args.results_dir,
        data_kind=args.data_kind,
    )
    return 0


if __name__ == "__main__":
    main()
