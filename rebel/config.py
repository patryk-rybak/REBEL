"""
Configuration file for REBEL framework.

Contains default parameters for attacks and model configurations.
"""

# Default attack parameters
NUM_ATTACKS = 100  # Number of attacks to perform

# Top-K list for sampling during evolutionary approach
TOP_K_LIST = [5, 10, 20, 50]

# Mutation strategies for evolutionary attack
MUTATIONS_LIST = [
    "synonym_replacement",
    "word_deletion",
    "word_insertion",
    "sentence_shuffle",
    "paraphrase"
]

# Hardcoded model configurations
# These are example model names - replace with actual models used in experiments
HARDCODED_MODELS = {
    "tofu": {
        "model_name": "locuslab/tofu_ft_llama2-7b",
        "tokenizer_name": "locuslab/tofu_ft_llama2-7b"
    },
    "wmdp": {
        "model_name": "cais/wmdp-llama2-7b",
        "tokenizer_name": "cais/wmdp-llama2-7b"
    },
    "default": {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "tokenizer_name": "meta-llama/Llama-2-7b-hf"
    }
}

# Evaluation parameters
BATCH_SIZE = 8
MAX_LENGTH = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# Evolutionary algorithm parameters
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7

# Logging
LOG_LEVEL = "INFO"
SAVE_RESULTS_PATH = "./results"
