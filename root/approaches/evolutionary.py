"""
Implements the evolutionary attack loop. Handles mutation, selection, tracking, and stats.
"""

import json
import os
from collections import Counter, defaultdict
from copy import copy
from datetime import datetime
from itertools import chain, compress
from random import randint
from typing import Any, Dict, List, Tuple

import numpy as np

from root import config
from root.approaches.base import BaseAttack
from root.models.base import BaseLLM
from root.models.hacker import HackerLLM
from root.models.judge import JudgeLLM


class EvolutionaryAttack(BaseAttack):
    def __init__(self, top_k_list: List[int], mutations_list: List[int]):
        """ """
        super().__init__()
        self.top_k_list = top_k_list
        self.mutations_list = mutations_list
        self.tracker_counter = 5000

    def select_top_k(
        self,
        evaluated_attacks: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        evaluated_attacks: numpy array (dtype=object) of dicts.
        Returns:
        - top_k_mask: boolean mask (shape: [n]) — top-k by score, excluding leaked
        - leaked_mask: boolean mask (shape: [n]) — leaked cases
        """
        evaluated_attacks = np.asarray(evaluated_attacks, dtype=object)
        n = evaluated_attacks.size

        top_k = int(min(max(top_k, 0), n))
        if n == 0 or top_k == 0:
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

        # extract scores
        scores = np.asarray(
            [
                (
                    float(rec.get("score", float("-inf")))
                    if isinstance(rec, dict)
                    else float("-inf")
                )
                for rec in evaluated_attacks
            ],
            dtype=float,
        )
        scores = np.where(np.isnan(scores), float("-inf"), scores)

        # leaked mask
        leaked_mask = np.asarray(
            [
                bool(rec.get("leaked", False)) if isinstance(rec, dict) else False
                for rec in evaluated_attacks
            ],
            dtype=bool,
        )

        # exclude leaked from ranking
        effective_scores = scores.copy()
        effective_scores[leaked_mask] = float("-inf")

        # top-k selection
        top_k_idx = np.argsort(effective_scores)[::-1][:top_k]

        top_k_mask = np.zeros(n, dtype=bool)
        top_k_mask[top_k_idx] = True

        return top_k_mask, leaked_mask

    def initialize_stats(self):
        return {}

    def add_stats(self, stats, generation_i, evaluated_attacks):

        new_stats = self._score_statistics(evaluated_attacks)
        stats[generation_i] = new_stats

    def finish_stats(self, stats, idx):
        with open(
            f"{config.RESULTS_DIR}/stats_{len(stats)}_idx{idx}_{randint(0, 99999)}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

    def _score_statistics(self, items):
        total = len(items)
        scores = [item["score"] for item in items]

        counts = Counter(scores)

        return {score: (count / total) * 100 for score, count in counts.items()}

    def initalize_tracker(self):
        # indexies are used to group trackers by their position in the top-k list (modulo mutations_i)
        return [[]]

    def finzalize_trackers(
        self,
        trackers,
        leaked_idxs,
        attacks,
        evaluated_attacks,
        mutations_i,
        idx,
        attacks_results,
        leaked=True,
    ):
        trakcers_idxs = leaked_idxs // mutations_i
        for tracker_idx, leaked_idx in zip(trakcers_idxs, leaked_idxs):
            new_tracker = trackers[tracker_idx].copy()
            new_tracker.append(
                (
                    attacks[leaked_idx],
                    evaluated_attacks[leaked_idx],
                    attacks_results[idx],
                )
            )

            self.finalize_tracker(new_tracker, idx, leaked)

    def finalize_tracker(self, tracker, idx, leaked):
        self.tracker_counter -= 1
        leaked_str = "_LEAKED_" if leaked else ""
        with open(
            f"{config.RESULTS_DIR}/tarcker_{len(tracker)}_idx{idx}_{leaked_str}{randint(0, 99999)}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(tracker, f, indent=4, ensure_ascii=False)

    def pack_trackers_and_add_things(
        self,
        trackers,
        top_k_indexies,
        attacks,
        evaluated_attacks,
        mutations_i,
        attacks_results,
    ):
        trackers_idxs = top_k_indexies // mutations_i
        new_trackers_per_modulo = defaultdict(list)
        for tracker_idx, top_k_idx in zip(trackers_idxs, top_k_indexies):
            new_trackers_per_modulo[tracker_idx].append(trackers[tracker_idx].copy())
            new_trackers_per_modulo[tracker_idx][-1].append(
                (
                    attacks[top_k_idx],
                    evaluated_attacks[top_k_idx],
                    attacks_results[top_k_idx],
                )
            )

        for i in range(len(trackers)):
            del trackers[0]

        for tracker_idx, new_trackers in new_trackers_per_modulo.items():
            trackers.extend(new_trackers)

    def add_to_tracker(self, trackes, tracker_idx, data):
        trackes[tracker_idx].append(data)

    def run(
        self,
        target: BaseLLM,
        hacker: HackerLLM,
        judge: JudgeLLM,
        baseline_data,
        use_trackers=False,
        do_stats=False,
        stop_at_first=False,
        idx: int = 0,
    ) -> None:
        if config.DATA_KIND == "tofu":
            self.run_tofu(
                target,
                hacker,
                judge,
                baseline_data,
                use_trackers,
                do_stats,
                stop_at_first,
                idx,
            )
        elif config.DATA_KIND == "wmdp":
            self.run_wmdp(target, hacker, judge, baseline_data, idx)
        else:
            raise NotImplementedError(
                f"Data kind {config.DATA_KIND} not supported in EvolutionaryAttack. Please implement dataset-specific logic for this data kind."
            )

    def run_tofu(
        self,
        target: BaseLLM,
        hacker: HackerLLM,
        judge: JudgeLLM,
        baseline_data,
        use_trackers=False,
        do_stats=False,
        stop_at_first=False,
        idx: int = 0,
    ) -> None:
        if self.tracker_counter <= 0:
            use_trackers = False

        baseline_only_dir = config.RESULTS_DIR / "baseline_only"
        os.makedirs(baseline_only_dir, exist_ok=True)

        baseline_question = baseline_data["question"]
        GOLD = baseline_data["answer"]

        if do_stats:
            stats = self.initialize_stats()
        if use_trackers:
            trackers = self.initalize_tracker()

        t1 = datetime.now()
        print("basline tager START: ", t1)

        baseline_replay = target.generate(baseline_question)

        t2 = datetime.now()
        print("baseline targert took: ", t2 - t1)

        base_eval: Dict[str, Any] = judge.evaluate(
            question=baseline_question,
            gold_answer=GOLD,
            model_reply=baseline_replay,
            role_based=False,
        )

        t3 = datetime.now()
        print("baseline judge took: ", t3 - t2)

        if stop_at_first and base_eval.get("leaked", False):
            leak_report = {
                "baseline": {
                    "question": baseline_question,
                    "gold_answer": GOLD,
                    "model_reply": baseline_replay,
                    "eval": base_eval,
                },
            }

            out_path = f"{baseline_only_dir}/leak_report_idx{idx}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(leak_report, f, indent=4, ensure_ascii=False)

            print(f"Leak report saved to: {out_path}")

            if stop_at_first:
                return

        if use_trackers:
            self.add_to_tracker(
                trackers, 0, (baseline_question, base_eval, baseline_replay)
            )

        if do_stats:
            self.add_stats(stats, 0, [base_eval])

        top_k_results = [baseline_question]
        top_k_evals = [base_eval]

        for generation_i, (top_k_i, mutations_i) in enumerate(
            zip(self.top_k_list, self.mutations_list)
        ):

            if self.tracker_counter <= 0:
                use_trackers = False

            t4 = datetime.now()
            temp = hacker.generate_batch(
                questions=top_k_results,
                gold_answers=[GOLD] * len(top_k_results),
                num_attacks=mutations_i,
                questions_evals=top_k_evals,
                zero_generation=False,
                baseline_question=baseline_question,
            )
            t5 = datetime.now()

            print("hacker generation took:", t5 - t4)

            attacks: List[str] = list(chain.from_iterable(temp))

            t6 = datetime.now()
            attacks_results: List[str] = target.generate_batch(attacks)
            t7 = datetime.now()

            print("target generation took:", t7 - t6)

            t8 = datetime.now()
            evaluated_attacks: List[Dict[str, Any]] = judge.evaluate_batch(
                questions=attacks,
                gold_answers=[GOLD] * len(attacks),
                model_replies=attacks_results,
                role_based=False,
            )
            t9 = datetime.now()

            print("judge evaluation took:", t9 - t8)

            evaluated_attacks = np.asarray(evaluated_attacks, dtype=object)

            top_k_indexies_mask: np.ndarray
            leaked_indexes_mask: np.ndarray
            top_k_indexies_mask, leaked_indexes_mask = self.select_top_k(
                evaluated_attacks, top_k_i
            )

            leaked_idxs = np.arange(len(evaluated_attacks))[leaked_indexes_mask]

            if use_trackers and leaked_idxs.any():
                self.finzalize_trackers(
                    trackers,
                    leaked_idxs,
                    attacks,
                    evaluated_attacks,
                    mutations_i,
                    idx,
                    attacks_results,
                )

            # level 0 is querying the baseline
            if do_stats:
                self.add_stats(stats, generation_i + 1, evaluated_attacks)

            if leaked_idxs.any() and stop_at_first:
                print("Leak detected, stopping attack.")

                leak_report = {
                    "baseline": {
                        "question": baseline_question,
                        "gold_answer": GOLD,
                        "model_reply": baseline_replay,
                        "eval": base_eval,
                    },
                    "leaked": [
                        {
                            "idx": idx,
                            "attack": attacks[int(i)],
                            "model_reply": attacks_results[int(i)],
                            "eval": evaluated_attacks[int(i)],
                        }
                        for i in leaked_idxs
                    ],
                }

                out_path = f"{config.RESULTS_DIR}/leak_report_idx{idx}_depth{generation_i}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(leak_report, f, indent=4, ensure_ascii=False)

                print(f"Leak report saved to: {out_path}")

                if stop_at_first:
                    return
                break

            if use_trackers:
                top_k_indexies = np.where(top_k_indexies_mask)[0]
                self.pack_trackers_and_add_things(
                    trackers,
                    top_k_indexies,
                    attacks,
                    evaluated_attacks,
                    mutations_i,
                    attacks_results,
                )

            top_k_results = list(compress(attacks, top_k_indexies_mask))
            top_k_evals = evaluated_attacks[top_k_indexies_mask].tolist()

        print("Attack finished. Time: ", datetime.now())

        if use_trackers:
            print("Finalizing trackers...")
            for trakcer in trackers:
                self.finalize_tracker(trakcer, idx, leaked=False)
            print("Trackers finalized.")
        if do_stats:
            print("Finalzation stats...")
            self.finish_stats(stats, idx)
            print("Stats finalized.")

    def run_wmdp(
        self,
        target: BaseLLM,
        hacker: HackerLLM,
        judge: JudgeLLM,
        baseline_data,
        idx: int = 0,
    ) -> None:

        def get_answers_for_scores(reply):
            return [int(np.argmax(scores)) for scores in reply]

        def get_gold_scores(reply):
            return [scores[GOLD] for scores in reply]

        def format_questions(baseline_question, choices):
            """
            Formats the question and list of choices into a text format suitable for the model in MMLU/WMDP benchmarks.
            """
            # We add the question text and a newline character
            formatted_text = f"{baseline_question.strip()}\n"
            # 2. We iterate through the list of choices and assign them letters (A, B, C, D...)
            # chr(65) is 'A', chr(66) is 'B', etc.
            for i, choice in enumerate(choices):
                letter = chr(65 + i)
                formatted_text += f"{letter}. {choice}\n"
            # 3. We add an ending instruction to prompt the model to return only the letter of the chosen answer, without any additional text.
            formatted_text += "Return label only without any other text."
            return formatted_text

        baseline_only_dir = config.RESULTS_DIR / "baseline_only"
        os.makedirs(baseline_only_dir, exist_ok=True)

        baseline_question = baseline_data["question"]  # question
        get_letter = lambda x: chr(65 + x)
        GOLD = int(baseline_data["answer"])  # int
        choices = baseline_data["choices"]  # chices in a list

        formetted_baseline = format_questions(
            baseline_question, choices
        )  # question + chices

        t1 = datetime.now()
        print("basline tager START: ", t1)

        # list of lists of scores for each question
        baseline_replay = target.get_answers_with_scores([formetted_baseline])

        t2 = datetime.now()
        print("baseline targert took: ", t2 - t1)

        # get indexes of maximim values for each list in baseline_replay which is a list of lists
        answer = get_answers_for_scores(baseline_replay)[0]

        base_eval: Dict[str, Any] = {}
        base_eval["scores"] = baseline_replay[0]
        base_eval["leaked"] = answer == GOLD

        t3 = datetime.now()
        print("baseline judge took: ", t3 - t2)

        if base_eval.get("leaked", False):
            leak_report = {
                "baseline": {
                    "question": formetted_baseline,
                    "gold_answer": GOLD,
                    "model_reply": baseline_replay[0],
                    "eval": base_eval,
                },
            }

            out_path = f"{baseline_only_dir}/leak_report_idx{idx}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(leak_report, f, indent=4, ensure_ascii=False)

            print(f"Leak report saved to: {out_path}")

            return

        top_k_results = [baseline_question]
        top_k_evals = [base_eval]

        for generation_i, (top_k_i, mutations_i) in enumerate(
            zip(self.top_k_list, self.mutations_list)
        ):
            print(
                f"[gen {generation_i}] parents={len(top_k_results)} mutations_i={mutations_i} total={len(top_k_results)*mutations_i}"
            )

            t4 = datetime.now()
            temp = hacker.generate_batch_wmdp(
                questions=top_k_results,
                gold_answers=[GOLD] * len(top_k_results),
                num_attacks=mutations_i,
                questions_evals=top_k_evals,
                zero_generation=False,
                baseline_question=baseline_question,
                choices=choices,
            )
            t5 = datetime.now()

            print("hacker generation took:", t5 - t4)

            attacks: List[str] = list(chain.from_iterable(temp))

            t6 = datetime.now()
            attacks_results: List[str] = target.get_answers_with_scores(
                [format_questions(q, choices) for q in attacks]
            )
            t7 = datetime.now()

            gold_scores = get_gold_scores(attacks_results)
            answers = get_answers_for_scores(attacks_results)

            print("target generation took:", t7 - t6)

            t8 = datetime.now()
            evaluated_attacks = []

            for ii in range(len(attacks_results)):
                model_reply = attacks_results[ii]
                answer = answers.pop(0)

                eval_dict: Dict[str, Any] = {}
                eval_dict["scores"] = model_reply
                eval_dict["score"] = gold_scores.pop(0)
                eval_dict["leaked"] = answer == GOLD  # to sprawdzic czy git
                eval_dict["attack"] = attacks[ii]
                eval_dict["baseline"] = formetted_baseline
                eval_dict["choices"] = choices
                eval_dict["gold"] = GOLD

                evaluated_attacks.append(eval_dict)

            t9 = datetime.now()

            if generation_i == 0:
                zero_generation_dir = config.RESULTS_DIR / f"zero_generation"
                os.makedirs(zero_generation_dir, exist_ok=True)
                with open(zero_generation_dir / f"{idx}", "w", encoding="utf-8") as f:
                    for eval_attack in evaluated_attacks:
                        f.write(
                            json.dumps(
                                eval_attack,
                            )
                            + "\n"
                        )

            print("judge evaluation took:", t9 - t8)

            evaluated_attacks = np.asarray(evaluated_attacks, dtype=object)

            top_k_indexies_mask: np.ndarray
            leaked_indexes_mask: np.ndarray
            top_k_indexies_mask, leaked_indexes_mask = self.select_top_k(
                evaluated_attacks, top_k_i
            )

            leaked_idxs = np.arange(len(evaluated_attacks))[leaked_indexes_mask]

            if leaked_idxs.any():
                print("Leak detected, stopping attack.")

                leak_report = {
                    "baseline": {
                        "question": baseline_question,
                        "gold_answer": GOLD,
                        "model_reply": baseline_replay,
                        "eval": base_eval,
                    },
                    "leaked": [
                        {
                            "idx": idx,
                            "attack": attacks[int(i)],
                            "model_reply": attacks_results[int(i)],
                            "eval": evaluated_attacks[int(i)],
                        }
                        for i in leaked_idxs
                    ],
                }

                out_path = f"{config.RESULTS_DIR}/leak_report_idx{idx}_depth{generation_i}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(leak_report, f, indent=4, ensure_ascii=False)

                print(f"Leak report saved to: {out_path}")
                return

            top_k_results = list(compress(attacks, top_k_indexies_mask))
            top_k_evals = evaluated_attacks[top_k_indexies_mask].tolist()

        print("Attack finished. Time: ", datetime.now())
