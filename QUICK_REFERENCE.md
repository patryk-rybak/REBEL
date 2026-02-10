# REBEL Quick Reference

## CLI Commands

### Naive Attack (leak@n)
```bash
# Basic usage with defaults
python main.py leak

# Custom number of attacks
python main.py leak --num-attacks 50

# Custom model
python main.py leak --no-hardcoded --model-name "your/model" --tokenizer-name "your/tokenizer"

# With debug logging
python main.py leak --num-attacks 100 --log-level DEBUG
```

### REBEL Evolutionary Attack
```bash
# Basic usage with defaults
python main.py rebel

# Custom parameters
python main.py rebel --num-attacks 200 --top-k 5 10 20 --mutations synonym_replacement paraphrase

# Custom model
python main.py rebel --no-hardcoded --model-name "your/model" --tokenizer-name "your/tokenizer"
```

## Project Structure Quick Reference

```
rebel/
├── attacks/
│   ├── naive_attack.py     # Implement: run_leak_attack()
│   └── rebel_attack.py     # Implement: run_rebel()
├── models/
│   └── loader.py           # Implement: load_model_and_tokenizer()
├── utils/
│   └── prompting.py        # Add your utility functions
├── evaluation/
│   └── metrics.py          # Implement: calculate_asr() and other metrics
└── config.py               # Update with your actual values
```

## Configuration Quick Reference

Edit `rebel/config.py`:

```python
# Attack parameters
NUM_ATTACKS = 100                    # Your default
TOP_K_LIST = [5, 10, 20, 50]       # Your values
MUTATIONS_LIST = ["your", "list"]   # Your mutations

# Model configurations
HARDCODED_MODELS = {
    "default": {
        "model_name": "your/model",
        "tokenizer_name": "your/tokenizer"
    }
}
```

## Migration Checklist

- [ ] Update `rebel/config.py` with actual defaults
- [ ] Migrate naive attack code to `rebel/attacks/naive_attack.py`
- [ ] Migrate REBEL code to `rebel/attacks/rebel_attack.py`
- [ ] Migrate model loading to `rebel/models/loader.py`
- [ ] Move utility functions to `rebel/utils/`
- [ ] Move evaluation metrics to `rebel/evaluation/`
- [ ] Test: `python main.py leak --num-attacks 5`
- [ ] Test: `python main.py rebel --num-attacks 5`
- [ ] Update model names in config to actual paths

## Help

```bash
python main.py --help           # General help
python main.py leak --help      # Naive attack help
python main.py rebel --help     # REBEL attack help
```

## Files to Read

1. [GETTING_STARTED.md](GETTING_STARTED.md) - Start here
2. [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Detailed migration steps
3. [README.md](README.md) - Full documentation
4. [examples/usage_example.py](examples/usage_example.py) - Code examples
