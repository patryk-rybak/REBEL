"""
Example: Running REBEL attacks programmatically

This example shows how to use REBEL's attack functions directly from Python code
rather than through the CLI.
"""

import sys
import os

# Add parent directory to path so we can import rebel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rebel import config
from rebel.attacks.naive_attack import run_leak_attack
from rebel.attacks.rebel_attack import run_rebel


def example_naive_attack():
    """Example of running a naive leak@n attack."""
    print("=" * 60)
    print("Example: Naive Attack (leak@n)")
    print("=" * 60)
    
    # Get default model from config
    model_config = config.HARDCODED_MODELS["default"]
    
    # Run attack
    results = run_leak_attack(
        model_name=model_config["model_name"],
        tokenizer_name=model_config["tokenizer_name"],
        num_attacks=10
    )
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print()


def example_rebel_attack():
    """Example of running a REBEL evolutionary attack."""
    print("=" * 60)
    print("Example: REBEL Evolutionary Attack")
    print("=" * 60)
    
    # Get default model from config
    model_config = config.HARDCODED_MODELS["default"]
    
    # Run attack with custom parameters
    results = run_rebel(
        model_name=model_config["model_name"],
        tokenizer_name=model_config["tokenizer_name"],
        num_attacks=20,
        top_k_list=[5, 10, 20],
        mutations_list=["synonym_replacement", "paraphrase"]
    )
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print()


def example_custom_model():
    """Example of using a custom model."""
    print("=" * 60)
    print("Example: Custom Model")
    print("=" * 60)
    
    # Use a custom model (replace with your actual model)
    custom_model = "your-organization/your-model"
    custom_tokenizer = "your-organization/your-tokenizer"
    
    results = run_leak_attack(
        model_name=custom_model,
        tokenizer_name=custom_tokenizer,
        num_attacks=5
    )
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    print("\nREBEL Framework - Usage Examples")
    print("=" * 60)
    print("\nNote: These examples use template implementations.")
    print("Integrate your actual attack code for real results.\n")
    
    # Run examples
    example_naive_attack()
    example_rebel_attack()
    
    # Uncomment to test custom model
    # example_custom_model()
    
    print("\nFor CLI usage, see: python main.py --help")
    print("=" * 60)
