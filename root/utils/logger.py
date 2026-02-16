"""
Writes leak summaries and full attack dumps. Formats baseline and attack details
"""

import os
from datetime import datetime
from typing import Any, Dict, List


class AttackLogger:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def do_your_thing(
        self,
        index: str,
        QUESTION: str,
        GOLD_ANSWER: str,
        BASELINE_REPLY: str,
        leak_cases: List[Dict[str, Any]],
        num_leaks: int,
        ATTACK_REPLIES: List[tuple],
        base_eval: Dict[str, Any],
    ) -> None:

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.results_dir / f"{index}_leak_summary_{ts}.txt"
        lines: List[str] = []
        lines.append("=== Leak Detection Summary ===")
        lines.append(f"Timestamp: {ts}")
        lines.append("")
        lines.append(f"QUESTION: {QUESTION}")
        lines.append(f"GOLD (confidential): {GOLD_ANSWER}")
        lines.append("")
        lines.append("## Baseline")
        lines.append(f"Baseline reply:\n{BASELINE_REPLY}")
        lines.append(
            f"Baseline leaked?: {'YES' if base_eval.get('leaked') else 'NO'} (score={base_eval.get('score',0.0):.2f})"
        )
        if base_eval.get("rationale"):
            lines.append(f"Judge rationale: {base_eval['rationale']}")
        if base_eval.get("match_spans"):
            lines.append(f"Matched spans: {base_eval['match_spans']}")
        lines.append("Baseline prompt:")
        lines.append(QUESTION)
        lines.append("")
        lines.append("## All leaked prompts & answers (baseline and/or attacks)")
        if not leak_cases:
            lines.append("None detected.")
        else:
            for case in leak_cases:
                tag = (
                    "BASELINE"
                    if case["kind"] == "baseline"
                    else f"ATTACK #{case['index']}"
                )
                lines.append(f"\n--- {tag} ---")
                lines.append("Prompt:")
                lines.append(case["prompt"])
                lines.append("")
                lines.append("Leaked reply:")
                lines.append(case["reply"])
                lines.append(f"(judge_score={case['score']:.2f})")
                if case.get("rationale"):
                    lines.append(f"Judge rationale: {case['rationale']}")
                if case.get("match_spans"):
                    lines.append(f"Matched spans: {case['match_spans']}")

        lines.append("")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # print("== ======================")
        # print(f"== summary SAVED at: {summary_path}")

        if num_leaks == 0:
            all_attacks_path = (
                self.results_dir / f"{index}_all_attacks_no_leak_{ts}.txt"
            )
            dump: List[str] = []
            dump.append(
                "=== Adversarial Prompts & Model Replies (No Attack Leak Detected) ==="
            )
            dump.append(f"Timestamp: {ts}")
            dump.append("")
            dump.append("## Baseline (for context)")
            dump.append("Prompt:")
            dump.append(QUESTION)
            dump.append("")
            dump.append("Model reply:")
            dump.append(BASELINE_REPLY)
            dump.append("")
            dump.append("## Attacks")
            for i, (p, r) in enumerate(ATTACK_REPLIES, start=1):
                dump.append(f"\n--- ATTACK #{i} ---")
                dump.append("Prompt:")
                dump.append(p)
                dump.append("")
                dump.append("Model reply:")
                dump.append(r)
                dump.append("")
            with open(all_attacks_path, "w", encoding="utf-8") as f:
                f.write("\n".join(dump))
            # print(f"============================ no leaks ==========================")
            # print(f"Saved prompts & replies (no attack leaks) to: {all_attacks_path}")
