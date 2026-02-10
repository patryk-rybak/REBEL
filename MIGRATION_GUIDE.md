# Migration Guide from root_refactor to New Structure

This guide helps you migrate your existing code from the `root_refactor` directory to the new, clean project structure.

## Overview

The new structure organizes code into logical modules:
- `rebel/attacks/` - Attack implementations
- `rebel/models/` - Model loading and management
- `rebel/utils/` - Utility functions
- `rebel/evaluation/` - Evaluation metrics
- `rebel/config.py` - Configuration

## Step-by-Step Migration

### 1. Identify Your Current Code Structure

First, identify what you have in `root_refactor`:
```bash
cd root_refactor
find . -name "*.py" -type f
```

### 2. Categorize Your Files

Categorize each Python file based on its purpose:
- **Attack code** (leak@n, REBEL implementation) → `rebel/attacks/`
- **Model code** (loading, tokenization) → `rebel/models/`
- **Helper functions** → `rebel/utils/`
- **Evaluation metrics** → `rebel/evaluation/`
- **Configuration variables** → update `rebel/config.py`

### 3. Move Attack Implementations

If you have files like `naive_attack.py` or `rebel_attack.py`:

```bash
# Example structure
cp root_refactor/naive_attack.py rebel/attacks/naive_attack.py
cp root_refactor/rebel_attack.py rebel/attacks/rebel_attack.py
cp root_refactor/evolutionary.py rebel/attacks/evolutionary.py
```

Then update imports in these files:
```python
# Old import
from config import NUM_ATTACKS

# New import
from rebel.config import NUM_ATTACKS
```

### 4. Move Model Code

If you have model-related code:

```bash
cp root_refactor/model_loader.py rebel/models/loader.py
cp root_refactor/tokenizer_utils.py rebel/models/tokenizer.py
```

### 5. Move Utility Functions

```bash
cp root_refactor/utils.py rebel/utils/helpers.py
cp root_refactor/prompting.py rebel/utils/prompting.py
```

### 6. Update Configuration

Review your existing configuration and update `rebel/config.py`:

```python
# Add your specific configurations
NUM_ATTACKS = 100  # Your default value
TOP_K_LIST = [5, 10, 20, 50]  # Your values
MUTATIONS_LIST = ["your", "mutation", "strategies"]

# Update hardcoded models with your actual model paths
HARDCODED_MODELS = {
    "tofu": {
        "model_name": "your-actual-tofu-model",
        "tokenizer_name": "your-actual-tofu-tokenizer"
    },
    # ... add more
}
```

### 7. Integrate with main.py

Update the TODO sections in `main.py` to call your migrated functions:

```python
def run_naive_attack(...):
    # Replace TODO with:
    from rebel.attacks.naive_attack import run_leak_attack
    
    results = run_leak_attack(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        num_attacks=num_attacks
    )
    return results

def run_rebel_attack(...):
    # Replace TODO with:
    from rebel.attacks.rebel_attack import run_rebel
    
    results = run_rebel(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        num_attacks=num_attacks,
        top_k_list=top_k_list,
        mutations_list=mutations_list
    )
    return results
```

### 8. Update All Imports

Search and replace old imports throughout your migrated files:

```bash
# Find files with old imports
grep -r "from root" rebel/
grep -r "import root" rebel/

# Update them to new structure
# from root_refactor.something import X
# ↓
# from rebel.something import X
```

### 9. Test the Migration

1. Test that imports work:
```bash
python -c "from rebel import config; print(config.NUM_ATTACKS)"
```

2. Test the CLI help:
```bash
python main.py --help
python main.py leak --help
python main.py rebel --help
```

3. Test with a simple run (won't work fully until you integrate your code):
```bash
python main.py leak --num-attacks 10 --log-level DEBUG
```

## Common Issues and Solutions

### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'rebel'`

**Solution:** Make sure you're running from the project root and have installed the package:
```bash
cd /path/to/REBEL
pip install -e .
```

### Issue: Circular Imports

**Problem:** Circular import dependencies

**Solution:** Reorganize code to avoid circular dependencies. Use:
- Dependency injection
- Import inside functions (not at module level)
- Separate interface from implementation

### Issue: Hardcoded Paths

**Problem:** Old code has hardcoded paths to `root_refactor/`

**Solution:** Use relative imports and the config module:
```python
# Bad
data_path = "root_refactor/data/dataset.json"

# Good
from rebel.config import SAVE_RESULTS_PATH
import os
data_path = os.path.join(SAVE_RESULTS_PATH, "dataset.json")
```

## Example: Complete Migration

Here's an example of migrating a simple naive attack:

### Before (root_refactor/naive.py):
```python
from config import NUM_ATTACKS
from model_utils import load_model

def run_naive():
    model = load_model("hardcoded-path")
    # attack logic
    pass
```

### After (rebel/attacks/naive_attack.py):
```python
from rebel.config import NUM_ATTACKS
from rebel.models.loader import load_model
from typing import Optional

def run_leak_attack(
    model_name: str,
    tokenizer_name: str,
    num_attacks: int = NUM_ATTACKS
) -> dict:
    """Run naive leak@n attack."""
    model = load_model(model_name)
    # attack logic
    return {"success": True, "results": ...}
```

## Checklist

After migration, verify:

- [ ] All Python files have been moved to appropriate directories
- [ ] All imports have been updated to use `rebel.*` structure
- [ ] No references to `root_refactor` remain in code
- [ ] Configuration is centralized in `rebel/config.py`
- [ ] `main.py` has been updated to call your functions
- [ ] The CLI works: `python main.py leak --help`
- [ ] Basic test runs without errors
- [ ] Added `root_refactor/` to `.gitignore`

## Need Help?

If you encounter issues during migration:
1. Check this guide
2. Review the example code in `rebel/` directories
3. Look at similar open-source projects for patterns
4. Test incrementally - migrate and test one module at a time

Good luck with your migration!
