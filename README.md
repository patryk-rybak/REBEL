# REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop

**Authors:** Patryk Rybak, Paweł Batorski, Paul Swoboda, Przemysław Spurek

## Abstract

Machine unlearning for LLMs aims to remove sensitive or copyrighted data from trained models. However, the true efficacy of current unlearning methods remains uncertain. Standard evaluation metrics rely on benign queries that often mistake superficial information suppression for genuine knowledge removal. Such metrics fail to detect residual knowledge that more sophisticated prompting strategies could still extract.

We introduce **REBEL**, an evolutionary approach for adversarial prompt generation designed to probe whether unlearned data can still be recovered. Our experiments demonstrate that REBEL successfully elicits “forgotten” knowledge from models that appeared to have forgotten it under standard unlearning benchmarks, revealing that current unlearning methods may provide only a superficial layer of protection.

We validate our framework on subsets of the **TOFU** and **WMDP** benchmarks, evaluating performance across a diverse suite of unlearning algorithms. Our experiments show that REBEL consistently outperforms static baselines, recovering “forgotten” knowledge with Attack Success Rates (ASRs) reaching up to **60% on TOFU** and **93% on WMDP**.

## Code Availability

The source code will be uploaded within two weeks.
