# REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop

**Authors:** Patryk Rybak, Paweł Batorski, Paul Swoboda, Przemysław Spurek

## Abstract

Machine unlearning for LLMs aims to remove sensitive or copyrighted data from trained models. However, the true efficacy of current unlearning methods remains uncertain. Standard evaluation metrics rely on benign queries that often mistake superficial information suppression for genuine knowledge removal. Such metrics fail to detect residual knowledge that more sophisticated prompting strategies could still extract.

We introduce **REBEL**, an evolutionary approach for adversarial prompt generation designed to probe whether unlearned data can still be recovered. Our experiments demonstrate that REBEL successfully elicits "forgotten" knowledge from models that appeared to have forgotten it under standard unlearning benchmarks, revealing that current unlearning methods may provide only a superficial layer of protection.

We validate our framework on subsets of the **TOFU** and **WMDP** benchmarks, evaluating performance across a diverse suite of unlearning algorithms. Our experiments show that REBEL consistently outperforms static baselines, recovering "forgotten" knowledge with Attack Success Rates (ASRs) reaching up to **60% on TOFU** and **93% on WMDP**.

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for running LLMs)

### Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/patryk-rybak/REBEL.git
cd REBEL
\`\`\`

2. Create a virtual environment (recommended):
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

Or install as a package:
\`\`\`bash
pip install -e .
\`\`\`

## Project Structure

\`\`\`
REBEL/
├── rebel/               # Main package directory
│   ├── attacks/         # Attack implementations (naive & REBEL)
│   ├── models/          # Model loading and management
│   ├── utils/           # Utility functions
│   ├── evaluation/      # Evaluation metrics
│   └── config.py        # Default configuration
├── main.py              # Main entry point with CLI
├── requirements.txt     # Python dependencies
├── setup.py            # Package setup
├── MIGRATION_GUIDE.md  # Guide for migrating existing code
└── README.md           # This file
\`\`\`

## Usage

REBEL provides two main attack modes via a command-line interface:
1. **leak** - Naive leak@n attack approach
2. **rebel** - Evolutionary REBEL attack approach

### Quick Start

Run naive attack with default parameters:
\`\`\`bash
python main.py leak --num-attacks 100
\`\`\`

Run REBEL evolutionary attack:
\`\`\`bash
python main.py rebel
\`\`\`

### Command-Line Interface

#### Common Arguments
- \`--log-level\`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

#### Naive Attack (leak@n)

\`\`\`bash
python main.py leak [OPTIONS]
\`\`\`

Options:
- \`--num-attacks\`: Number of attacks to perform (default: 100)
- \`--hardcoded\`: Use hardcoded model configurations (default: True)
- \`--no-hardcoded\`: Use custom model names
- \`--model-name\`: Custom model name from HuggingFace
- \`--tokenizer-name\`: Custom tokenizer name from HuggingFace

Example:
\`\`\`bash
# Run with custom model
python main.py leak --num-attacks 50 \
    --no-hardcoded \
    --model-name "meta-llama/Llama-2-7b-hf" \
    --tokenizer-name "meta-llama/Llama-2-7b-hf"
\`\`\`

#### REBEL Evolutionary Attack

\`\`\`bash
python main.py rebel [OPTIONS]
\`\`\`

Options:
- \`--num-attacks\`: Number of attacks to perform (default: 100)
- \`--top-k\`: Top-K values for sampling (default: [5, 10, 20, 50])
- \`--mutations\`: Mutation strategies to use
- \`--hardcoded\`: Use hardcoded model configurations (default: True)
- \`--no-hardcoded\`: Use custom model names
- \`--model-name\`: Custom model name from HuggingFace
- \`--tokenizer-name\`: Custom tokenizer name from HuggingFace

Example:
\`\`\`bash
# Run with custom parameters
python main.py rebel \
    --num-attacks 200 \
    --top-k 5 10 20 \
    --mutations synonym_replacement word_deletion paraphrase
\`\`\`

### Configuration

Default parameters are defined in \`rebel/config.py\`:
- \`NUM_ATTACKS\`: Default number of attacks (100)
- \`TOP_K_LIST\`: Default top-k values ([5, 10, 20, 50])
- \`MUTATIONS_LIST\`: Default mutation strategies
- \`HARDCODED_MODELS\`: Pre-configured model names for different benchmarks

You can modify these defaults or override them via command-line arguments.

## Migration from Existing Code

If you have existing code in a \`root_refactor\` or similar directory, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions on:
- How to organize your code into the new structure
- Where to place different types of files
- How to update imports
- How to integrate with the new CLI

## Development Status

⚠️ **Note**: The framework structure is complete, but attack implementations need to be integrated. The current code provides:
- ✅ Clean, organized project structure
- ✅ Command-line interface with argparse
- ✅ Configuration management
- ✅ Template files for attack implementations
- ⏳ Actual attack algorithms (to be integrated from existing code)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

\`\`\`bibtex
@article{rybak2024rebel,
  title={REBEL: Hidden Knowledge Recovery via Evolutionary-Based Evaluation Loop},
  author={Rybak, Patryk and Batorski, Paweł and Swoboda, Paul and Spurek, Przemysław},
  journal={arXiv preprint},
  year={2024}
}
\`\`\`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
