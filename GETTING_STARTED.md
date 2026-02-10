# Getting Started with REBEL

This guide will help you get up and running with the REBEL framework.

## What Has Been Done

The project has been restructured with a clean, modular architecture:

### ✅ Completed

1. **Clean Project Structure**
   - `rebel/` package with submodules for attacks, models, utils, and evaluation
   - Separation of concerns with clear module boundaries
   - Template files ready for your existing code

2. **Command-Line Interface**
   - Full argparse-based CLI in `main.py`
   - Support for `leak` (naive) and `rebel` (evolutionary) attack modes
   - Configurable parameters via command-line arguments
   - Flag for hardcoded vs custom models

3. **Configuration Management**
   - Centralized configuration in `rebel/config.py`
   - Default values for NUM_ATTACKS, TOP_K_LIST, MUTATIONS_LIST
   - Hardcoded model configurations (ready to be updated with your actual models)

4. **Documentation**
   - Updated README with usage instructions
   - MIGRATION_GUIDE.md with step-by-step migration instructions
   - This getting started guide

### ⏳ To Be Done (By You)

The framework is ready, but you need to integrate your existing attack implementations:

1. **Migrate your code** from `root_refactor` to the new structure
2. **Implement the attack logic** in the template files
3. **Update config.py** with your actual model names
4. **Test** the integrated code

## Next Steps

### Step 1: Test the CLI (Optional)

Before migrating code, test that the CLI works:

```bash
# Show help
python main.py --help

# Test leak command
python main.py leak --help
python main.py leak --num-attacks 10

# Test rebel command
python main.py rebel --help
python main.py rebel --num-attacks 5 --top-k 5 10
```

You'll see warnings that implementations are pending - that's expected!

### Step 2: Identify Your Existing Code

Navigate to your `root_refactor` directory and identify:

1. **Attack code** - Where is your naive attack (leak@n) implementation?
2. **REBEL code** - Where is your evolutionary attack implementation?
3. **Model code** - How do you load models and tokenizers?
4. **Utilities** - What helper functions do you use?
5. **Config** - What are your actual default values?

### Step 3: Update Configuration

Edit `rebel/config.py` with your actual values:

```python
# Update these with your actual defaults
NUM_ATTACKS = 100  # Your default
TOP_K_LIST = [5, 10, 20, 50]  # Your values
MUTATIONS_LIST = ["your", "actual", "mutations"]

# Update with your actual model names
HARDCODED_MODELS = {
    "tofu": {
        "model_name": "path/to/your/tofu/model",
        "tokenizer_name": "path/to/your/tofu/tokenizer"
    },
    "wmdp": {
        "model_name": "path/to/your/wmdp/model",
        "tokenizer_name": "path/to/your/wmdp/tokenizer"
    },
    # ... add more as needed
}
```

### Step 4: Migrate Attack Code

#### For Naive Attack

1. Find your naive attack implementation in `root_refactor`
2. Copy the logic to `rebel/attacks/naive_attack.py`
3. Update the `run_leak_attack` function
4. Update imports to use the new structure

Example:
```python
# In rebel/attacks/naive_attack.py
def run_leak_attack(
    model_name: str,
    tokenizer_name: str,
    num_attacks: int = NUM_ATTACKS,
    **kwargs
) -> Dict[str, Any]:
    # Import your dependencies
    from rebel.models.loader import load_model_and_tokenizer
    from rebel.evaluation.metrics import calculate_asr
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name)
    
    # YOUR EXISTING ATTACK CODE HERE
    # ...
    
    return {
        "attack_type": "naive_leak",
        "num_attacks": num_attacks,
        "asr": asr_score,
        "results": results
    }
```

#### For REBEL Attack

1. Find your REBEL implementation in `root_refactor`
2. Copy the logic to `rebel/attacks/rebel_attack.py`
3. Update the `run_rebel` function
4. Update imports

### Step 5: Migrate Model Loading

If you have custom model loading logic:

1. Update `rebel/models/loader.py` with your implementation
2. Remove the TODO and implement `load_model_and_tokenizer`

Example:
```python
# In rebel/models/loader.py
def load_model_and_tokenizer(model_name, tokenizer_name=None, device="cuda", **kwargs):
    from transformers import AutoModel, AutoTokenizer
    
    if tokenizer_name is None:
        tokenizer_name = model_name
    
    model = AutoModel.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = model.to(device)
    
    return model, tokenizer
```

### Step 6: Migrate Utilities

Copy any utility functions you have to the appropriate module:
- Prompting utilities → `rebel/utils/prompting.py`
- Evaluation metrics → `rebel/evaluation/metrics.py`
- Other helpers → create new files in `rebel/utils/`

### Step 7: Test Your Integration

After integrating your code, test it:

```bash
# Test with a small number of attacks first
python main.py leak --num-attacks 5 --log-level DEBUG

# If that works, try REBEL
python main.py rebel --num-attacks 5 --log-level DEBUG

# Test with custom models
python main.py leak --no-hardcoded \
    --model-name "your/model" \
    --tokenizer-name "your/tokenizer" \
    --num-attacks 10
```

### Step 8: Update .gitignore

Make sure your backup directories are ignored:

```bash
# The .gitignore already includes:
root/
root_refactor/
*_backup/
```

## Common Integration Patterns

### Pattern 1: Simple Function Migration

If your code is organized as functions:

```python
# Old: root_refactor/naive.py
def do_naive_attack(model, n):
    # ... your code

# New: rebel/attacks/naive_attack.py
from rebel.models.loader import load_model_and_tokenizer

def run_leak_attack(model_name, tokenizer_name, num_attacks, **kwargs):
    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name)
    # Use your existing do_naive_attack logic
    return results
```

### Pattern 2: Class-Based Migration

If your code uses classes:

```python
# Old: root_refactor/rebel_class.py
class RebelAttacker:
    def __init__(self, model):
        # ...
    
    def attack(self):
        # ...

# New: rebel/attacks/rebel_attack.py
# Keep your class, just update imports
# Use it in run_rebel function
def run_rebel(...):
    model, tokenizer = load_model_and_tokenizer(...)
    attacker = RebelAttacker(model)
    results = attacker.attack()
    return results
```

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'rebel'`:

```bash
# Make sure you're in the project root
cd /path/to/REBEL

# Install in development mode
pip install -e .
```

### Path Issues

If you have hardcoded paths in your old code:

```python
# Bad (old code)
data = load("root_refactor/data/file.json")

# Good (new code)
import os
from rebel.config import SAVE_RESULTS_PATH
data = load(os.path.join(SAVE_RESULTS_PATH, "file.json"))
```

### Model Loading Issues

If models won't load:

1. Check that model names in config.py are correct
2. Ensure you have access to the models (HuggingFace token, etc.)
3. Check GPU availability
4. Try with `--log-level DEBUG` for more information

## Need More Help?

- See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration steps
- Check [README.md](README.md) for CLI usage
- Review template files in `rebel/` for expected signatures

## Summary

You now have:
✅ A clean, professional project structure
✅ A working command-line interface
✅ Configuration management
✅ Template files ready for your code
✅ Documentation and migration guides

Next: Integrate your existing attack code and test!

Good luck! 🚀
